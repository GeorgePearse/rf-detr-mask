"""
Tests for per-class COCO metrics functionality.
"""

import unittest
from unittest.mock import Mock
import numpy as np
import pandas as pd

from rfdetr.util.per_class_metrics import (
    print_per_class_metrics,
    get_coco_category_names,
    get_per_class_metrics_dataframe,
)


class TestPerClassMetrics(unittest.TestCase):
    """Test per-class metrics functions."""

    def setUp(self):
        """Set up mock COCO evaluator and ground truth."""
        # Create mock COCO ground truth
        self.mock_coco_gt = Mock()
        self.mock_coco_gt.cats = {
            1: {"name": "person"},
            2: {"name": "bicycle"},
            3: {"name": "car"},
            4: {"name": "motorcycle"},
            5: {"name": "airplane"},
        }

        # Create mock COCO evaluator
        self.mock_evaluator = Mock()
        self.mock_evaluator.coco_eval = {}

        # Create mock evaluation results
        mock_eval = Mock()
        mock_eval.params = Mock()
        mock_eval.params.catIds = [1, 2, 3, 4, 5]
        mock_eval.params.iouThrs = np.array(
            [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        )
        mock_eval.params.maxDets = [1, 10, 100]

        # Create mock precision and recall arrays
        # Shape: (num_iou_thresholds, num_recall_thresholds, num_categories, num_area_ranges, num_max_dets)
        num_iou = 10
        num_recall = 101
        num_cats = 5
        num_areas = 4
        num_max_dets = 3

        # Generate some realistic-looking AP values
        np.random.seed(42)
        precision = np.random.rand(
            num_iou, num_recall, num_cats, num_areas, num_max_dets
        )
        # Make some values -1 (no detections)
        precision[precision < 0.1] = -1

        recall = np.random.rand(num_iou, num_cats, num_areas, num_max_dets)
        recall[recall < 0.1] = -1

        mock_eval.eval = {
            "precision": precision,
            "recall": recall,
        }

        self.mock_evaluator.coco_eval["bbox"] = mock_eval

    def test_get_coco_category_names(self):
        """Test extracting category names from COCO ground truth."""
        names = get_coco_category_names(self.mock_coco_gt)

        self.assertEqual(len(names), 5)
        self.assertEqual(names[1], "person")
        self.assertEqual(names[3], "car")

    def test_get_coco_category_names_no_cats(self):
        """Test with no categories."""
        mock_gt = Mock()
        mock_gt.cats = {}
        names = get_coco_category_names(mock_gt)
        self.assertEqual(names, {})

    def test_get_per_class_metrics_dataframe(self):
        """Test getting per-class metrics as DataFrame."""
        df = get_per_class_metrics_dataframe(
            self.mock_evaluator,
            iou_type="bbox",
            class_names={
                1: "person",
                2: "bicycle",
                3: "car",
                4: "motorcycle",
                5: "airplane",
            },
            metric_name="AP",
            iou_threshold=None,
            max_dets=100,
            area_range="all",
        )

        # Check DataFrame structure
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn("class_id", df.columns)
        self.assertIn("class_name", df.columns)
        self.assertIn("AP", df.columns)

        # Check that it's sorted by AP (descending)
        if len(df) > 1:
            self.assertTrue((df["AP"].values[:-1] >= df["AP"].values[1:]).all())

    def test_get_per_class_metrics_dataframe_specific_iou(self):
        """Test getting metrics at specific IoU threshold."""
        df = get_per_class_metrics_dataframe(
            self.mock_evaluator,
            iou_type="bbox",
            class_names={
                1: "person",
                2: "bicycle",
                3: "car",
                4: "motorcycle",
                5: "airplane",
            },
            metric_name="AP",
            iou_threshold=0.5,
            max_dets=100,
            area_range="all",
        )

        self.assertIsInstance(df, pd.DataFrame)
        # Should have valid results
        self.assertGreaterEqual(len(df), 0)

    def test_print_per_class_metrics(self):
        """Test printing per-class metrics."""
        # This mainly tests that the function runs without errors
        # Since it prints to stdout, we're just checking it doesn't crash
        result = print_per_class_metrics(
            self.mock_evaluator,
            iou_type="bbox",
            class_names={
                1: "person",
                2: "bicycle",
                3: "car",
                4: "motorcycle",
                5: "airplane",
            },
            metric_name="AP",
            iou_threshold=None,
            max_dets=100,
            area_range="all",
        )

        # Should return a dictionary
        self.assertIsInstance(result, dict)
        # Keys should be category IDs
        for key in result.keys():
            self.assertIn(key, [1, 2, 3, 4, 5])

    def test_invalid_iou_type(self):
        """Test with invalid IoU type."""
        df = get_per_class_metrics_dataframe(
            self.mock_evaluator,
            iou_type="invalid",
            class_names={1: "person"},
            metric_name="AP",
        )

        # Should return empty DataFrame
        self.assertTrue(df.empty)

    def test_invalid_area_range(self):
        """Test with invalid area range."""
        df = get_per_class_metrics_dataframe(
            self.mock_evaluator,
            iou_type="bbox",
            class_names={1: "person"},
            metric_name="AP",
            area_range="invalid",
        )

        # Should return empty DataFrame
        self.assertTrue(df.empty)

    def test_ar_metric(self):
        """Test Average Recall metric."""
        df = get_per_class_metrics_dataframe(
            self.mock_evaluator,
            iou_type="bbox",
            class_names={
                1: "person",
                2: "bicycle",
                3: "car",
                4: "motorcycle",
                5: "airplane",
            },
            metric_name="AR",
            iou_threshold=None,
            max_dets=100,
            area_range="all",
        )

        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn("AR", df.columns)


if __name__ == "__main__":
    unittest.main()
