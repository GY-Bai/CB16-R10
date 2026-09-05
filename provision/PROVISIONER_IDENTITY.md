# Provisioner identity

The Python environment cache identity includes the implementation bytes of the Python environment resolver/provisioner, in addition to runtime identity, environment manifest, and requirements bytes.

This prevents a cached environment created under older installer/index semantics from being silently reused after changes to `provision/scripts/provision_python.py` or `provision/scripts/resolve_environment.py`.

The resulting `environment_sha256` remains the canonical Python environment identity used by CI evidence.
