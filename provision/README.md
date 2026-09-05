# CB16 Dependency & Asset Provisioner V1

Declarative, cache-first provisioner for Python environments and external assets on the Shanxi worker.

## Layout

```text
provision/
├── schemas/
├── environments/
├── assets/
├── providers/
└── scripts/
```

Environments are selected by CI profile. Assets are referenced by logical `asset_id` and never committed as binaries.

## Run

```bash
python provision/scripts/provision_all.py --profile smoke --job-id test
```

Outputs are written under `CB16_CI_WORKER_ROOT/jobs/<job_id>/`.
