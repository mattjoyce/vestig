# Test Document for Chunk Provenance

This is a test document to verify that chunk provenance tracking is working correctly.

## Section 1: Background

The chunk provenance feature tracks the exact location in source documents where each memory was extracted from. This enables tracing memories back to their original source.

## Section 2: Implementation

The implementation stores chunk references in the format "path:start:length" in the metadata field of each memory. This simple text format can later be parsed to create proper FILE and CHUNK nodes when migrating to Neo4j.

## Section 3: Benefits

Key benefits include:
- Full provenance tracking without schema changes
- Ability to re-extract entities from original chunks
- Support for future Neo4j migration
- Better debugging and transparency
