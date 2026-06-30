"""
pytest configuration for SAHA tests.
Enables asyncio mode and sets up common fixtures.
"""
import pytest


# All async tests use pytest-asyncio automatically (asyncio_mode = "auto" in pyproject.toml)
