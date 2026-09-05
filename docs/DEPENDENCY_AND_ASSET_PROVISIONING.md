# CB16 Dependency & Asset Provisioning

## Adding a Python package

1. Add/update `requirements/...` or environment manifest under `provision/environments/<profile>.json`.
2. Commit only declaration and lock info.
3. Provisioner creates/reuses a hashed venv under `/data/cb16_ci/venvs/<hash>`.

## Adding a model / asset

1. Add `provision/assets/<asset_id>.json`.
2. Declare `asset_id`, `type`, `storage.logical_path`, and `sources`.
3. If a canonical SHA256 is known use `integrity_mode=FROZEN`; otherwise use `DISCOVERY`.
4. Never commit binaries or secrets.

## Adding a dataset

If the dataset already exists on Shanxi, register a `manual_existing` source and optionally a local hint env. The Provisioner will register the real local path without re-downloading.

## When download source is unknown

Use `DISCOVERY` mode and multiple candidate `sources`. The Provisioner will try adapters in order and emit `DISCOVERED_ASSET.json`; it will not freeze the asset as scientific authority until a later authority commit pins SHA256.
