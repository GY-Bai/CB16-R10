"""Provider adapter registry for CB16 Provisioner V1."""
from .base import Candidate, Provider, ProviderError
from .http import HttpProvider
from .huggingface import HuggingFaceProvider
from .local import LocalProvider
from .modelscope import ModelScopeProvider
from .oci_cache import OciCacheProvider

PROVIDERS = [
    LocalProvider(),
    HttpProvider(),
    HuggingFaceProvider(),
    ModelScopeProvider(),
    OciCacheProvider(),
]
