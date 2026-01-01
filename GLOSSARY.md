# Vestig Glossary

A concise glossary of core terms, mechanics, and maturity concepts used in Vestig.

## A
- **Artifact**: An input source used for ingestion (session transcript, JSONL export, notes, etc.).

## B
- **Bi-temporal model**: Separates event time (when a fact was true) from transaction time (when we learned it).
- **Boost / Reinforcement**: Signal that a memory re-occurred, used for temporal ranking rather than duplicating content.

## C
- **Chunking**: Splitting input text into manageable segments for extraction.
- **Commit**: The act of storing a memory (and related graph data) after hygiene and dedupe checks.
- **Content hash**: SHA-256 hash of normalized memory text used for exact dedupe.

## D
- **Daydream mode**: M6 concept for speculative synthesis; never auto-committed.
- **Dedupe (Exact / Near)**: Prevents duplicate memories; near-dupe detection treats similarity as reinforcement.

## E
- **Edge**: Relationship between nodes in the graph.
- **Edge invalidation**: Marking a prior edge as expired when contradicted by new info.
- **Embedding**: Vector representation of text used for semantic retrieval.
- **Entity**: Typed node extracted from memories (PERSON, ORG, SYSTEM, PROJECT, PLACE, SKILL, TOOL, FILE, CONCEPT).
- **Event node**: Append-only record of memory events (ADD, REINFORCE, IMPORT, DEPRECATE, SUMMARY_CREATED, etc.).

## G
- **Graph expansion**: Traversing edges (MENTIONS/RELATED) to find related nodes.

## H
- **HyDE (Hypothetical Document Embeddings)**: Generating hypothetical queries for better retrieval (M5+).
- **Hygiene (Quality Firewall)**: Input filtering rules to block low-signal or noisy memories (M2).

## I
- **Ingestion**: Pipeline that reads an artifact, extracts memories, and commits them.

## K
- **Kind (MEMORY / SUMMARY)**: Discriminator on memory nodes that separates primary memories from derived summary nodes.

## L
- **Learning lag**: Time between when a fact was true (t_valid) and when we learned it (t_created).
- **Lateral thinking**: M6 retrieval mode to surface non-obvious associations.

## M
- **Maturity levels (M1–M6)**: Progressive capability slices from core loop to advanced cognition.
- **Memory**: Canonical stored fact/insight with metadata and embeddings.
- **MemRank**: Graph centrality score (PageRank-like) used in advanced retrieval (M5).
- **MENTIONS**: Edge type for Memory → Entity links.

## N
- **Norm key**: Normalized entity key for deduplication (type + canonicalized name).

## P
- **Progressive maturation**: Build the smallest end-to-end slice, lock interfaces, then deepen.

## Q
- **Quality Firewall**: See Hygiene; protects the corpus from low-value content.

## R
- **Recall**: Retrieval formatted for agent context insertion.
- **RELATED**: Edge type for Memory → Memory semantic similarity links.

## S
- **Source**: Metadata describing how a memory was created (manual, hook, import, ingest).
- **Substance filter**: LLM gate that rejects low-signal content before storage (M2+).
- **Summary generation**: LLM step that produces a SUMMARY node after an ingest run (currently when >=5 memories are committed).
- **Summary node**: Derived memory node (kind=SUMMARY) that synthesizes an ingest run and links to its source memories.
- **SUMMARIZES**: Edge type from Summary → Memory that records which memories were summarized.

## T
- **Temporal decay**: Recency weighting for memory relevance (M3).
- **Temporal stability**: Classification of facts as static/dynamic/unknown.
- **TraceRank**: Temporal reinforcement ranking using event history (M3).
- **Transaction time (t_created)**: When the system learned the fact.

## V
- **Validity time (t_valid / t_invalid)**: When a fact became true / stopped being true (event time).

## W
- **Working set**: M6 short-term memory for active contexts.
