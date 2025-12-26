"""
Agent Memory System
Combines SQLite storage with PyDatalog querying for LLM agent memory
"""

import sqlite3
import json
import hashlib
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import os
import pickle
import llm

# Note: PyDatalog would be used here, but for demonstration we'll show the structure
# pip install pyDatalog would be needed in production

class AgentMemory:
    """Memory system for LLM agents using SQLite + Datalog"""
    
    def __init__(self, db_path: str = "agent_memory.db"):
        self.db_path = db_path
        self.conn = None
        self._initialize_database()
    
    def _initialize_database(self):
        """Create the database schema"""
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()
        
        # Core facts table (EAV with metadata)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT,
                value_type TEXT,  -- string|number|date|reference|boolean
                confidence REAL DEFAULT 1.0,
                source_session TEXT,
                timestamp TEXT,
                context TEXT,
                extraction_method TEXT DEFAULT 'naive',  -- naive|discerned
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(subject, predicate, object, source_session)
            )
        """)
        
        # Sessions metadata
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                project TEXT,
                cwd TEXT,
                start_time TEXT,
                end_time TEXT,
                total_messages INTEGER,
                user_messages INTEGER,
                assistant_messages INTEGER,
                tools_used TEXT,  -- JSON array
                summary TEXT,
                file_hash TEXT,  -- SHA256 hash for deduplication
                file_path TEXT,  -- Original file path
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Skills/MCP servers registry
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS skills (
                skill_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                skill_type TEXT,  -- mcp|native|custom
                mcp_server_url TEXT,
                parameters TEXT,  -- JSON
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Embeddings (for vector search integration)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_id TEXT NOT NULL,
                entity_type TEXT,  -- fact|session|predicate
                text_content TEXT,  -- The text that was embedded
                embedding BLOB,  -- Pickled numpy array or list
                model TEXT,  -- Model used for embedding
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(entity_id, entity_type, model)
            )
        """)
        
        # Create indexes for common queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(subject)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_facts_predicate ON facts(predicate)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_facts_session ON facts(source_session)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_facts_timestamp ON facts(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_facts_extraction_method ON facts(extraction_method)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_hash ON sessions(file_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_entity ON embeddings(entity_id, entity_type)")

        self.conn.commit()
    
    def is_session_imported(self, session_id: str = None, file_hash: str = None) -> bool:
        """Check if session was already imported by ID or file hash"""
        cursor = self.conn.cursor()

        if file_hash:
            cursor.execute("SELECT 1 FROM sessions WHERE file_hash = ?", (file_hash,))
            if cursor.fetchone():
                return True

        if session_id:
            cursor.execute("SELECT 1 FROM sessions WHERE session_id = ?", (session_id,))
            if cursor.fetchone():
                return True

        return False

    def compute_file_hash(self, filepath: str) -> str:
        """Compute SHA256 hash of a file for deduplication"""
        sha256 = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    def import_session(self, session_data: Dict[str, Any], file_path: str = None,
                      skip_if_exists: bool = True):
        """
        Import parsed session data into the database

        Args:
            session_data: Parsed session from ClaudeCodeSessionParser
            file_path: Original file path (for hash computation)
            skip_if_exists: Skip import if session already exists
        """
        cursor = self.conn.cursor()

        # Compute file hash for deduplication
        file_hash = None
        if file_path and os.path.exists(file_path):
            file_hash = self.compute_file_hash(file_path)

            if skip_if_exists and self.is_session_imported(file_hash=file_hash):
                print(f"Session already imported (hash: {file_hash[:16]}...), skipping")
                return False

        # Check by session_id as fallback
        session_id = session_data['metadata'].get('session_id')
        if skip_if_exists and self.is_session_imported(session_id=session_id):
            print(f"Session {session_id} already imported, skipping")
            return False

        # Import session metadata
        meta = session_data['metadata']
        cursor.execute("""
            INSERT OR REPLACE INTO sessions
            (session_id, project, cwd, start_time, end_time,
             total_messages, user_messages, assistant_messages, tools_used, summary,
             file_hash, file_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            meta['session_id'],
            meta.get('project'),
            meta.get('cwd'),
            meta.get('start_time'),
            meta.get('end_time'),
            meta.get('total_messages'),
            meta.get('user_messages'),
            meta.get('assistant_messages'),
            json.dumps(meta.get('tools_used', [])),
            session_data.get('summary'),
            file_hash,
            file_path
        ))
        
        # Import facts
        for fact in session_data['facts']:
            try:
                cursor.execute("""
                    INSERT OR IGNORE INTO facts
                    (subject, predicate, object, value_type, confidence,
                     source_session, timestamp, context, extraction_method)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    fact['subject'],
                    fact['predicate'],
                    fact['object'],
                    fact['value_type'],
                    fact['confidence'],
                    fact['source_session'],
                    fact['timestamp'],
                    fact.get('context', ''),
                    fact.get('extraction_method', 'naive')
                ))
            except sqlite3.IntegrityError:
                # Duplicate fact, skip
                pass

        self.conn.commit()
        print(f"✓ Imported session {meta['session_id']} with {len(session_data['facts'])} facts")
        return True
    
    def query_facts(self, subject: Optional[str] = None, 
                   predicate: Optional[str] = None,
                   object_value: Optional[str] = None,
                   session: Optional[str] = None,
                   min_confidence: float = 0.0) -> List[Dict]:
        """Query facts with optional filters"""
        cursor = self.conn.cursor()
        
        query = "SELECT * FROM facts WHERE confidence >= ?"
        params = [min_confidence]
        
        if subject:
            query += " AND subject = ?"
            params.append(subject)
        
        if predicate:
            query += " AND predicate = ?"
            params.append(predicate)
        
        if object_value:
            query += " AND object = ?"
            params.append(object_value)
        
        if session:
            query += " AND source_session = ?"
            params.append(session)
        
        query += " ORDER BY timestamp DESC"
        
        cursor.execute(query, params)
        columns = [desc[0] for desc in cursor.description]
        
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def find_related_sessions(self, criteria: Dict[str, Any], limit: int = 10) -> List[Dict]:
        """Find sessions matching criteria"""
        cursor = self.conn.cursor()
        
        query = "SELECT * FROM sessions WHERE 1=1"
        params = []
        
        if criteria.get('project'):
            query += " AND project = ?"
            params.append(criteria['project'])
        
        if criteria.get('tools_used'):
            # JSON search (basic, would need better JSON support in production)
            for tool in criteria['tools_used']:
                query += " AND tools_used LIKE ?"
                params.append(f'%{tool}%')
        
        query += " ORDER BY start_time DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        columns = [desc[0] for desc in cursor.description]
        
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def get_entity_knowledge(self, entity: str) -> Dict[str, Any]:
        """Get all facts about a specific entity"""
        facts = self.query_facts(subject=entity)
        
        # Organize by predicate
        knowledge = {'entity': entity, 'facts': {}}
        for fact in facts:
            pred = fact['predicate']
            if pred not in knowledge['facts']:
                knowledge['facts'][pred] = []
            knowledge['facts'][pred].append({
                'value': fact['object'],
                'confidence': fact['confidence'],
                'source': fact['source_session'],
                'timestamp': fact['timestamp']
            })
        
        return knowledge
    
    def find_patterns(self, pattern_type: str = 'task_pattern') -> List[Dict]:
        """Find extracted patterns"""
        return self.query_facts(predicate=pattern_type)
    
    def get_discoveries(self, min_confidence: float = 0.6) -> List[Dict]:
        """Get all discoveries/findings"""
        return self.query_facts(predicate='discovery', min_confidence=min_confidence)
    
    def get_solutions(self, min_confidence: float = 0.6) -> List[Dict]:
        """Get all solutions found"""
        return self.query_facts(predicate='provides_solution', min_confidence=min_confidence)
    
    def export_to_datalog(self, output_file: str):
        """Export facts in Datalog format for use with PyDatalog"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT subject, predicate, object FROM facts WHERE confidence >= 0.7")
        
        with open(output_file, 'w') as f:
            f.write("% Facts extracted from agent sessions\n\n")
            
            for subject, predicate, obj in cursor.fetchall():
                # Clean values for Datalog syntax
                subj_clean = subject.replace('-', '_').replace('.', '_')
                pred_clean = predicate.replace('-', '_')
                obj_clean = obj.replace("'", "\\'") if obj else ""
                
                f.write(f"+{pred_clean}('{subj_clean}', '{obj_clean}')\n")
        
        print(f"Exported facts to {output_file}")
    
    def stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        cursor = self.conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM facts")
        total_facts = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM sessions")
        total_sessions = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT subject) FROM facts")
        unique_entities = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT predicate) FROM facts")
        unique_predicates = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT predicate, COUNT(*) as count 
            FROM facts 
            GROUP BY predicate 
            ORDER BY count DESC 
            LIMIT 10
        """)
        top_predicates = dict(cursor.fetchall())
        
        return {
            'total_facts': total_facts,
            'total_sessions': total_sessions,
            'unique_entities': unique_entities,
            'unique_predicates': unique_predicates,
            'top_predicates': top_predicates
        }
    
    def generate_embedding(self, text: str, model_name: str = "ada-002") -> Optional[List[float]]:
        """
        Generate embedding using llm Python API

        Args:
            text: Text to embed
            model_name: Embedding model (default: ada-002 for OpenAI)

        Returns:
            List of floats representing the embedding
        """
        try:
            # Get embedding model
            model = llm.get_embedding_model(model_name)

            # Generate embedding
            embedding = model.embed(text)

            # Convert to list if needed
            if hasattr(embedding, 'tolist'):
                return embedding.tolist()
            return list(embedding)

        except Exception as e:
            print(f"Embedding error: {e}")
            return None

    def embed_fact(self, fact_id: int, force: bool = False, model: str = "ada-002"):
        """
        Generate and store embedding for a single fact

        Args:
            fact_id: Database ID of the fact
            force: Re-generate even if embedding exists
            model: Embedding model to use
        """
        cursor = self.conn.cursor()

        # Check if embedding already exists
        if not force:
            cursor.execute(
                "SELECT 1 FROM embeddings WHERE entity_id = ? AND entity_type = 'fact' AND model = ?",
                (str(fact_id), model)
            )
            if cursor.fetchone():
                return  # Already embedded

        # Get fact details
        cursor.execute("SELECT subject, predicate, object, context FROM facts WHERE id = ?", (fact_id,))
        row = cursor.fetchone()
        if not row:
            return

        subject, predicate, obj, context = row

        # Create text representation for embedding
        # Format: "predicate: object [context]"
        text_parts = [f"{predicate}: {obj}"]
        if context:
            text_parts.append(f"[{context[:200]}]")

        text_content = " ".join(text_parts)

        # Generate embedding
        embedding = self.generate_embedding(text_content, model=model)
        if not embedding:
            return

        # Store embedding
        embedding_blob = pickle.dumps(embedding)
        cursor.execute("""
            INSERT OR REPLACE INTO embeddings
            (entity_id, entity_type, text_content, embedding, model)
            VALUES (?, 'fact', ?, ?, ?)
        """, (str(fact_id), text_content, embedding_blob, model))

        self.conn.commit()

    def embed_all_facts(self, model: str = "ada-002", batch_size: int = 10):
        """
        Generate embeddings for all facts that don't have them

        Args:
            model: Embedding model to use
            batch_size: Number of facts to process before committing
        """
        cursor = self.conn.cursor()

        # Find facts without embeddings
        cursor.execute("""
            SELECT f.id
            FROM facts f
            LEFT JOIN embeddings e ON f.id = CAST(e.entity_id AS INTEGER)
                AND e.entity_type = 'fact'
                AND e.model = ?
            WHERE e.id IS NULL
        """, (model,))

        fact_ids = [row[0] for row in cursor.fetchall()]

        if not fact_ids:
            print("All facts already have embeddings")
            return

        print(f"Generating embeddings for {len(fact_ids)} facts...")

        for i, fact_id in enumerate(fact_ids, 1):
            self.embed_fact(fact_id, model=model)

            if i % batch_size == 0:
                print(f"  Processed {i}/{len(fact_ids)} facts")

        print(f"✓ Generated {len(fact_ids)} embeddings")

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors"""
        a = np.array(vec1)
        b = np.array(vec2)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def find_similar_facts(self, query_text: str, top_k: int = 5,
                          model: str = "ada-002",
                          min_confidence: float = 0.0,
                          extraction_method: str = None) -> List[Dict[str, Any]]:
        """
        Find facts semantically similar to query text (RAG retrieval)

        Args:
            query_text: Text to find similar facts for
            top_k: Number of results to return
            model: Embedding model to use
            min_confidence: Minimum fact confidence
            extraction_method: Filter by 'naive' or 'discerned'

        Returns:
            List of facts with similarity scores
        """
        # Generate embedding for query
        query_embedding = self.generate_embedding(query_text, model=model)
        if not query_embedding:
            print("Failed to generate query embedding")
            return []

        # Get all embedded facts
        cursor = self.conn.cursor()

        query = """
            SELECT f.*, e.embedding, e.text_content
            FROM facts f
            JOIN embeddings e ON f.id = CAST(e.entity_id AS INTEGER)
            WHERE e.entity_type = 'fact'
              AND e.model = ?
              AND f.confidence >= ?
        """
        params = [model, min_confidence]

        if extraction_method:
            query += " AND f.extraction_method = ?"
            params.append(extraction_method)

        cursor.execute(query, params)

        results = []
        for row in cursor.fetchall():
            # Unpack row (facts table has 11 columns + 2 from embeddings)
            fact_id, subject, predicate, obj, value_type, confidence, session, timestamp, context, extr_method, created_at, embedding_blob, text_content = row

            # Deserialize embedding
            embedding = pickle.loads(embedding_blob)

            # Compute similarity
            similarity = self.cosine_similarity(query_embedding, embedding)

            results.append({
                'fact_id': fact_id,
                'subject': subject,
                'predicate': predicate,
                'object': obj,
                'confidence': confidence,
                'extraction_method': extr_method,
                'source_session': session,
                'timestamp': timestamp,
                'context': context,
                'similarity': similarity,
                'embedded_text': text_content
            })

        # Sort by similarity and return top-k
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


# Example Datalog rules (would be in PyDatalog in production)
EXAMPLE_DATALOG_RULES = """
# PyDatalog rules for agent memory reasoning

from pyDatalog import pyDatalog

# Define terms
pyDatalog.create_terms('used_tool, executed_command, connects_to_host')
pyDatalog.create_terms('mentions_path, targets_system, provides_solution')
pyDatalog.create_terms('Task, Action, System, Path, Solution')
pyDatalog.create_terms('can_solve, requires_system, worked_on_project')

# Facts would be loaded from database
# +used_tool('action_123', 'Bash')
# +connects_to_host('action_123', '192.168.20.4')
# +targets_system('task_456', 'unraid')

# Rules for inference
# Can solve a problem if we have a solution mentioning the same system
can_solve(Task, Solution) <= (
    targets_system(Task, System) &
    provides_solution(Solution, _) &
    mentions_system(Solution, System)
)

# Actions that worked on a specific system
worked_on_system(Action, System) <= (
    connects_to_host(Action, Host) &
    system_host(System, Host)
)

# Find related tasks by common tools
related_tasks(Task1, Task2) <= (
    used_tool(Task1, Tool) &
    used_tool(Task2, Tool) &
    (Task1 != Task2)
)

# Example queries:
# can_solve('new_unraid_task', Solution)?
# worked_on_system(Action, 'unraid')?
# related_tasks('task_A', X)?
"""


def main():
    """Example usage"""
    import sys
    import argparse

    parser_cli = argparse.ArgumentParser(description='Import sessions into agent memory')
    parser_cli.add_argument('session_file', nargs='?', help='Path to session JSONL file')
    parser_cli.add_argument('--discerned', action='store_true',
                           help='Enable LLM-based discerned extraction')
    parser_cli.add_argument('--model', default='claude-3-5-haiku-20241022',
                           help='LLM model for discerned extraction')
    parser_cli.add_argument('--db', default='example_agent_memory.db',
                           help='Database path')

    args = parser_cli.parse_args()

    # Initialize memory system
    memory = AgentMemory(args.db)

    # Import session if provided
    if args.session_file:
        from session_parser import ClaudeCodeSessionParser

        parser = ClaudeCodeSessionParser(
            use_discerned=args.discerned,
            llm_model=args.model
        )
        session_data = parser.parse_session_file(args.session_file)
        memory.import_session(session_data, file_path=args.session_file)
    
    # Show stats
    print("\n=== Memory Statistics ===")
    stats = memory.stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Example queries
    print("\n=== Example Queries ===")
    
    print("\n1. Sessions in 'unraid_admin' project:")
    sessions = memory.find_related_sessions({'project': 'unraid_admin'})
    for session in sessions:
        print(f"  - {session['session_id']}: {session['summary']}")
    
    print("\n2. All discoveries:")
    discoveries = memory.get_discoveries()
    for disc in discoveries[:3]:
        print(f"  - {disc['object'][:100]}... (confidence: {disc['confidence']})")
    
    print("\n3. Solutions found:")
    solutions = memory.get_solutions()
    for sol in solutions[:3]:
        print(f"  - {sol['object'][:100]}... (confidence: {sol['confidence']})")
    
    # Export to Datalog format
    print("\n=== Exporting to Datalog ===")
    memory.export_to_datalog("agent_memory.datalog")
    
    memory.close()
    
    # Show example Datalog rules
    print("\n=== Example Datalog Rules ===")
    print(EXAMPLE_DATALOG_RULES)


if __name__ == '__main__':
    main()
