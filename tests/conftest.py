"""
Pytest configuration and fixtures for RF-DETR-Mask tests.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock


@pytest.fixture
def device():
    """Get the appropriate device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_batch():
    """Create a sample batch of images for testing."""
    batch_size = 2
    channels = 3
    height, width = 224, 224

    images = torch.randn(batch_size, channels, height, width)
    targets = [
        {
            "labels": torch.randint(0, 80, (5,)),
            "boxes": torch.rand(5, 4),
            "image_id": torch.tensor([i]),
            "orig_size": torch.tensor([height, width]),
            "size": torch.tensor([height, width]),
        }
        for i in range(batch_size)
    ]

    return images, targets


@pytest.fixture
def mock_coco_evaluator():
    """Create a mock COCO evaluator for testing metrics."""
    evaluator = Mock()
    evaluator.coco_eval = {}

    # Create mock evaluation results
    mock_eval = Mock()
    mock_eval.params = Mock()
    mock_eval.params.catIds = list(range(1, 81))  # 80 COCO categories
    mock_eval.params.iouThrs = np.linspace(0.5, 0.95, 10)
    mock_eval.params.maxDets = [1, 10, 100]

    # Generate mock precision/recall arrays
    num_iou = 10
    num_recall = 101
    num_cats = 80
    num_areas = 4
    num_max_dets = 3

    np.random.seed(42)
    precision = np.random.rand(num_iou, num_recall, num_cats, num_areas, num_max_dets)
    recall = np.random.rand(num_iou, num_cats, num_areas, num_max_dets)

    mock_eval.eval = {
        "precision": precision,
        "recall": recall,
    }

    evaluator.coco_eval["bbox"] = mock_eval
    return evaluator


@pytest.fixture
def minimal_args():
    """Create minimal arguments for testing."""

    class Args:
        # Model parameters
        num_classes = 80
        num_queries = 900
        hidden_dim = 256
        position_embedding = "sine"

        # Training parameters
        batch_size = 2
        lr = 1e-4
        weight_decay = 1e-4
        epochs = 1
        clip_max_norm = 0.1

        # Loss parameters
        set_cost_class = 5
        set_cost_bbox = 2
        set_cost_giou = 1
        cls_loss_coef = 5.0
        bbox_loss_coef = 2.0
        giou_loss_coef = 1.0

        # Other parameters
        device = "cpu"
        amp = False
        distributed = False

    return Args()


# Skip slow tests by default
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
