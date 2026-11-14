"""
Unit Tests for Stage 3 Generalization & Stage 4 Autonomous Triage Meta-Predictor in federated learning.
"""

import os
import numpy as np
import pytest
import torch
import torchvision.transforms as transforms
from torchvision.datasets import FakeData

from fl_outage.data.scientific_partitions import (
    CorruptedDataset,
    create_scientific_partition_indices,
    get_scientific_subsets
)
from fl_outage.models.vit import ViTTinyLN
from analysis.triage_decision_tree import (
    PureNumPyDecisionTree,
    determine_optimal_action,
    compile_meta_predictor_dataset,
    evaluate_cross_validation,
    FEATURE_NAMES,
    ACTION_NAMES
)


def test_feature_skew_corrupted_dataset():
    """Verifies that CorruptedDataset produces valid, altered images for mild and severe corruptions."""
    fake_data = FakeData(size=20, image_size=(3, 32, 32), num_classes=10, transform=transforms.ToTensor())
    
    clean_img, _ = fake_data[0]
    mild_ds = CorruptedDataset(fake_data, corruption_level="mild")
    severe_ds = CorruptedDataset(fake_data, corruption_level="severe")

    mild_img, _ = mild_ds[0]
    severe_img, _ = severe_ds[0]

    assert clean_img.shape == (3, 32, 32)
    assert mild_img.shape == (3, 32, 32)
    assert severe_img.shape == (3, 32, 32)

    # Corruptions should introduce variation in pixel values
    assert not torch.allclose(clean_img, mild_img, atol=1e-3)
    assert not torch.allclose(clean_img, severe_img, atol=1e-3)


def test_feature_skew_partition_generation():
    """Verifies that R3-F feature skew uses balanced disjoint IID halves."""
    targets = np.repeat(np.arange(10), 100)  # Exactly 100 samples per class
    idx_A, idx_B = create_scientific_partition_indices(targets, source="R3-F", seed=42)

    assert len(idx_A) == 500
    assert len(idx_B) == 500
    assert len(set(idx_A).intersection(set(idx_B))) == 0  # Disjoint


def test_vit_tiny_ln_architecture():
    """Verifies ViTTinyLN forward pass, feature extraction, and parameter count."""
    model = ViTTinyLN(
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dim=128,
        depth=4,
        num_heads=4,
        mlp_dim=128
    )

    x = torch.randn(4, 3, 32, 32)
    logits, feat = model(x, return_features=True)

    assert logits.shape == (4, 10)
    assert feat.shape == (4, 128)

    param_count = sum(p.numel() for p in model.parameters())
    # Should be well within lightweight bounds (~300k - 500k params)
    assert 200_000 <= param_count <= 550_000


def test_pure_numpy_decision_tree_separable():
    """Verifies CART tree achieves 100% accuracy on linearly separable data and exports valid ASCII rules."""
    np.random.seed(42)
    # 2 classes separable by feature 0
    X = np.array([
        [1.0, 5.0], [2.0, 6.0], [1.5, 4.0],
        [8.0, 1.0], [9.0, 2.0], [8.5, 0.5]
    ])
    y = np.array([0, 0, 0, 1, 1, 1])

    tree = PureNumPyDecisionTree(max_depth=2)
    tree.fit(X, y)
    preds = tree.predict(X)

    assert np.array_equal(preds, y)
    rules = tree.export_ascii_rules(["feat0", "feat1"], ["CLASS_A", "CLASS_B", "CLASS_C", "CLASS_D"])
    assert "RECOMMEND: CLASS_A" in rules
    assert "RECOMMEND: CLASS_B" in rules


def test_meta_predictor_cross_validation_and_dataset():
    """Verifies that Stage 2 dataset compiles correctly and achieves high cross-validation accuracy."""
    stage2_dir = "./runs/stage2_head_cures"
    if not os.path.exists(stage2_dir):
        pytest.skip("Stage 2 directory not found.")

    X, y, records = compile_meta_predictor_dataset(stage2_dir=stage2_dir)
    assert len(records) in (36, 54)
    assert X.shape[1] == 5
    assert len(y) == len(records)

    cv_acc = evaluate_cross_validation(X, y, k_folds=5, seed=42)
    assert cv_acc >= 0.90  # Decision boundary between onset <= 6% and > 6% should be distinct
