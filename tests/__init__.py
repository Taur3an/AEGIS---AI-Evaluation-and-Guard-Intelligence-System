"""
AEGIS Test Suite

Comprehensive test suite for the AEGIS AI Evaluation and Guard Intelligence System.
"""

# Test configuration
import pytest
import logging

# Configure logging for tests
logging.basicConfig(
    level=logging.WARNING,  # Reduce noise during tests
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Common test fixtures and utilities will be defined in conftest.py