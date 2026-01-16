# Vestig Modular Configuration

This directory contains modular, composable configuration files using the new `include` feature.

## Directory Structure

```
configs/
├── base/              # Core infrastructure configs
│   ├── storage-falkordb.yaml
│   └── embedding-gemma.yaml
├── models/            # LLM provider configs
│   ├── models-ollama-local.yaml
│   └── models-anthropic-saas.yaml
├── ontologies/        # Domain-specific entity ontologies
│   └── ontology-knowledge.yaml
└── composed/          # Complete configs for specific use cases
    ├── config-batch-knowledge.yaml
    └── config-interactive-knowledge.yaml
```

## Usage

### Batch Ingestion (Overnight, Local Ollama)

For slow, reliable overnight document ingestion using local Ollama models:

```bash
vestig --config configs/composed/config-batch-knowledge.yaml memory ingest docs/
```

**Features**:
- Uses local Ollama (qwen2.5:14b)
- No API costs
- Longer retention (365 days)
- Larger result sets for thoroughness

### Interactive Recall (Real-time, Anthropic SaaS)

For fast, interactive recall and ad-hoc memory queries using Anthropic API:

```bash
vestig --config configs/composed/config-interactive-knowledge.yaml memory recall "topic"
```

**Features**:
- Uses Anthropic Claude (fast, high quality)
- Optimized for speed
- Shorter retention (180 days)
- Focused result sets

## Include Syntax

Configs support the `include` key for composition:

```yaml
# Single include
include: base/storage-falkordb.yaml

# Multiple includes (merged in order)
include:
  - base/storage-falkordb.yaml
  - base/embedding-gemma.yaml
  - models/models-ollama-local.yaml

# Override included values
storage:
  falkordb:
    graph_name: custom-graph  # Overrides base config
```

### Include Behavior

1. **Include order matters**: Later includes override earlier ones
2. **Main config wins**: Values in the main config file always take final precedence
3. **Deep merge**: Nested dictionaries are merged recursively
4. **Relative paths**: Include paths are relative to the config file containing them
5. **Recursive**: Included files can themselves include other files

## Creating Custom Configs

### Example: Development Config

```yaml
# configs/composed/config-dev.yaml
include:
  - ../base/storage-falkordb.yaml
  - ../base/embedding-gemma.yaml
  - ../ontologies/ontology-minimal.yaml
  - ../models/models-ollama-local.yaml

storage:
  falkordb:
    graph_name: vestig-dev

hygiene:
  auto_expire_memories: false  # Keep everything during dev
```

### Example: Testing New Ontology

```yaml
# configs/ontologies/ontology-research-v2.yaml
m4:
  entity_types:
    - Person
    - Paper
    - Experiment
    - Dataset
    - Metric
  entity_extraction_mode: hybrid
```

Then create composed config:

```yaml
# configs/composed/config-research-v2-test.yaml
include:
  - ../base/storage-falkordb.yaml
  - ../base/embedding-gemma.yaml
  - ../ontologies/ontology-research-v2.yaml  # Test new ontology
  - ../models/models-anthropic-saas.yaml

storage:
  falkordb:
    graph_name: vestig-research-v2-test  # Separate graph for testing
```

## Benefits

1. **DRY**: Define once, reuse everywhere (storage, embedding, ontologies)
2. **Versioning**: Track ontology evolution (ontology-v1.yaml, ontology-v2.yaml)
3. **Testing**: Easy A/B testing of configurations
4. **Model Flexibility**: Swap between Ollama/Anthropic/other providers easily
5. **Domain Separation**: Different ontologies for different document types
6. **Cost Optimization**: Use cheap local models for batch, fast SaaS for interactive

## Migration from Flat Configs

Old flat config:

```yaml
# config.yaml
storage:
  falkordb:
    host: localhost
    port: 6379
    graph_name: vestig
embedding:
  model: embeddinggemma
  dimension: 768
m4:
  llm_provider: ollama
  model_name: qwen2.5:14b
  entity_types: [Person, Organization]
```

New modular equivalent:

```yaml
# configs/composed/my-config.yaml
include:
  - ../base/storage-falkordb.yaml
  - ../base/embedding-gemma.yaml
  - ../models/models-ollama-local.yaml

m4:
  entity_types: [Person, Organization]
```

## Tips

1. **Start simple**: Begin with a single include, then split as needed
2. **Semantic grouping**: Group by concern (storage, models, ontology)
3. **Version ontologies**: Use suffixes like `-v1`, `-v2` for experimentation
4. **Separate graphs**: Use different graph_name values for testing new configs
5. **Document overrides**: Comment why you're overriding included values
