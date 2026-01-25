# Claude Commands (Vestig)

This repo includes Claude command definitions used for Vestig recall and ingestion.
They live under `~/.claude/commands` and invoke the `vestig` CLI with a config file.

## Commands

### `vestig-context`
Recalls relevant memories from the Vestig graph:

```
vestig --config test/config-custom-ontology-falkordb.yaml memory recall '<args>'
```

### `vestig-remember`
Ingests the current conversation:

```
echo '{conversation}' | vestig --config test/config-custom-ontology-falkordb.yaml ingest - --format claude-session
```

## Adapting the config

Both commands take a `--config` path. To use a different config:

1. Pick the config you want to use (for example, `config.yaml` or a file under `configs/`).
2. Update the `--config` argument in each command file:
   - `~/.claude/commands/vestig-context.md`
   - `~/.claude/commands/vestig-remember.md`

Example swap:

```
--config config.yaml
```

Notes:
- Paths are resolved relative to your current working directory.
- If you frequently change configs, consider creating additional command variants
  (for example, `vestig-context-local.md`) with different `--config` values.
