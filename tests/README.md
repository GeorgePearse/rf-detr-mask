# RF-DETR-Mask Tests

This directory contains unit and integration tests for the RF-DETR-Mask project.

## Running Tests

### Using pytest (recommended)
```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ -v --cov=rfdetr --cov-report=term-missing

# Run specific test file
python -m pytest tests/test_per_class_metrics.py -v

# Run excluding slow tests
python -m pytest tests/ -v -m "not slow"
```

### Using unittest
```bash
# Run all tests
python -m unittest discover tests

# Run specific test file
python -m unittest tests.test_per_class_metrics
```

## Test Structure

- `test_per_class_metrics.py` - Tests for per-class COCO metrics functionality
- `test_model_loading.py` - Tests for model initialization and loading
- `test_training_utils.py` - Tests for training utilities and helper functions
- `conftest.py` - Pytest configuration and shared fixtures

## GitHub Actions

Tests are automatically run on:
- Every push to the main branch
- Every pull request

The CI pipeline includes:
- Linting with ruff
- Type checking with mypy
- Unit tests with pytest
- Code coverage reporting
- Security checks with bandit

## Writing New Tests

When adding new functionality, please include appropriate tests:

1. Create a new test file following the naming convention `test_*.py`
2. Use descriptive test names that explain what is being tested
3. Include both positive and negative test cases
4. Mock external dependencies when appropriate
5. Mark slow tests with `@pytest.mark.slow`
6. Mark GPU-requiring tests with `@pytest.mark.gpu`

Example test structure:
```python
import unittest
from unittest.mock import Mock, patch

class TestNewFeature(unittest.TestCase):
    def setUp(self):
        # Set up test fixtures
        pass
    
    def test_feature_normal_case(self):
        # Test normal operation
        pass
    
    def test_feature_edge_case(self):
        # Test edge cases
        pass
    
    def test_feature_error_handling(self):
        # Test error conditions
        pass
```