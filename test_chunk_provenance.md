# Test Document for Chunk Provenance

This is a test document to verify that chunk provenance tracking is working correctly.

## Section 1: Background

The chunk provenance feature tracks the exact location in source documents where each memory was extracted from. This enables tracing memories back to their original source.

## Section 2: Implementation

The implementation stores chunk nodes in FalkorDB and links Source → Chunk → Memory with edges. This keeps provenance explicit and queryable without string parsing.

## Section 3: Benefits

Key benefits include:
- Full provenance tracking with graph-native edges
- Ability to re-extract entities from original chunks
- Graph-first provenance ready for advanced traversal
- Better debugging and transparency
