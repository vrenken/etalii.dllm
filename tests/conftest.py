import os
import sys
from pathlib import Path

import pytest

# Lets tests import golden_values.py directly.
sys.path.insert(0, str(Path(__file__).parent))


@pytest.fixture
def cli_environment(monkeypatch):
    """No ``DLLM_*`` variables while the test runs, and none left behind: the CLI sets ``DLLM_MODEL`` and friends
    in ``os.environ`` directly, which would leak into later tests."""
    from etalii_dllm.engine import default_engine

    for name in [n for n in os.environ if n.startswith("DLLM_")]:
        monkeypatch.delenv(name)
    default_engine.cache_clear()
    yield monkeypatch
    for name in [n for n in os.environ if n.startswith("DLLM_")]:
        del os.environ[name]
    default_engine.cache_clear()
