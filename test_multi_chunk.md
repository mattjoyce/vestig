# Multi-Chunk Test Document

## Chapter 1: Introduction (Characters 0-1000)

This is the first chunk of a multi-chunk document designed to test chunk provenance tracking across multiple chunks. The chunk provenance feature tracks the exact location in source documents where each memory was extracted from. This enables tracing memories back to their original source with precise character offsets.

The implementation stores chunk references in the format "path:start:length" in the metadata field of each memory. This simple text format can later be parsed to create proper FILE and CHUNK nodes when migrating to Neo4j. This is a critical feature for maintaining data lineage and provenance.

Key aspects of the chunk provenance system include maintaining absolute file paths to ensure uniqueness across the filesystem, tracking both start position and length to enable precise extraction, storing in existing metadata fields to avoid schema changes, and supporting future migration to graph databases like Neo4j.

## Chapter 2: Technical Details (Characters 1000-2000)

The chunking algorithm works by dividing large documents into smaller segments with configurable overlap. Each chunk maintains its position metadata throughout the extraction pipeline. When memories are extracted from chunks, they inherit the chunk reference automatically.

The chunk reference format follows a simple pattern: full absolute path, colon, start character position, colon, length in characters. For example: "/path/to/file.md:1000:3500" indicates the chunk starts at position 1000 and is 3500 characters long.

This metadata is stored as JSON in the memories table, making it queryable via SQLite's json_extract function. The storage approach was chosen to minimize changes to the existing schema while maximizing future flexibility. When migrating to Neo4j, these strings can be parsed to create proper FILE and CHUNK node types with CONTAINS and EXTRACTED_FROM relationships.

## Chapter 3: Benefits and Use Cases (Characters 2000-3000)

The chunk provenance system provides several important benefits for knowledge management systems. First, it enables full traceability from any memory back to its exact source location. This is crucial for verification, auditing, and understanding context.

Second, it supports re-extraction of entities from original source text rather than from LLM-summarized memories. This improves entity extraction quality because entities are identified in their original context, not in potentially lossy summaries.

Third, the system facilitates debugging and quality assurance. When investigating memory quality issues, developers can quickly locate the source chunk and review the original text. This accelerates the feedback loop for improving extraction prompts and parameters.

Fourth, chunk provenance enables incremental document updates. By tracking which chunks have been processed, the system can efficiently re-process only changed sections of documents rather than re-ingesting entire files.

## Chapter 4: Future Enhancements (Characters 3000-4000)

Future enhancements to the chunk provenance system could include several advanced features. One possibility is implementing chunk-level versioning to track how source documents change over time. This would enable temporal queries like "show me all memories extracted from version 2 of this document."

Another enhancement could be bi-directional linking between chunks and memories, allowing efficient queries like "show all memories extracted from this chunk" or "show all chunks that contributed to this entity." This would require indexing the chunk_ref field for fast lookup.

A third enhancement could be chunk-level metadata enrichment, storing additional context like document section headings, document type, author, creation date, or modification timestamp. This would enable richer provenance queries and better context for memory retrieval.

Finally, the system could implement chunk deduplication detection, identifying when the same content appears at multiple locations across documents. This would help reduce redundant storage and improve cross-document entity linking.

## Chapter 5: Conclusion (Characters 4000+)

The chunk provenance tracking system represents an important step toward production-ready knowledge management. By maintaining precise source location references without requiring schema changes, it provides a lightweight yet powerful foundation for traceability and future enhancements.

The design balances immediate practical needs with long-term architectural goals. The simple string format is easy to implement and query, while remaining forward-compatible with more sophisticated graph database migrations planned for future releases.

This implementation demonstrates how thoughtful metadata design can unlock significant value without major system changes, providing a model for other incremental enhancements to the Vestig knowledge management platform.
