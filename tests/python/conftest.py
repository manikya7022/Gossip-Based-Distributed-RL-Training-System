"""Pytest configuration."""

import pytest


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")


@pytest.fixture
def sample_gradients():
    """Sample gradients for testing."""
    import numpy as np
    return np.random.randn(1000).astype(np.float32)
