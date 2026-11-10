import pytest
import torch
import torch.nn.functional as F
import numpy as np

from fl_outage.models.resnet import ResNet20LN
import importlib
outage_v2 = importlib.import_module("experiments.07_outage_v2")
stage_of = outage_v2.stage_of
make_splits = outage_v2.make_splits
Pipe = outage_v2.Pipe


def test_parameter_stage_mapping_resnet20ln():
    """Verify that stage_of covers 100% of parameters in ResNet20LN and stage 0 has no stages.* parameters."""
    model = ResNet20LN(num_classes=10)
    sd = model.state_dict()
    nb = len(model.stages)  # 9 blocks for ResNet-20

    stages_assigned = {}
    for name in sd.keys():
        st = stage_of(name, nb)
        assert 0 <= st <= 4, f"Parameter {name} assigned invalid stage {st}"
        stages_assigned[name] = st

    # Stage 0 must contain only init_conv and init_norm (stem), no stages.* parameters
    stage_0_params = [k for k, v in stages_assigned.items() if v == 0]
    assert all("stages." not in k and "fc." not in k for k in stage_0_params)
    assert any("init_conv" in k for k in stage_0_params)
    assert any("init_norm" in k for k in stage_0_params)

    # Stages 1, 2, 3 must contain residual stages
    stage_1_params = [k for k, v in stages_assigned.items() if v == 1]
    stage_2_params = [k for k, v in stages_assigned.items() if v == 2]
    stage_3_params = [k for k, v in stages_assigned.items() if v == 3]

    assert all(k.startswith("stages.0.") or k.startswith("stages.1.") or k.startswith("stages.2.") for k in stage_1_params)
    assert all(k.startswith("stages.3.") or k.startswith("stages.4.") or k.startswith("stages.5.") for k in stage_2_params)
    assert all(k.startswith("stages.6.") or k.startswith("stages.7.") or k.startswith("stages.8.") for k in stage_3_params)

    # Stage 4 must contain only the classification head
    stage_4_params = [k for k, v in stages_assigned.items() if v == 4]
    assert set(stage_4_params) == {"fc.weight", "fc.bias"}


def test_make_splits_pairwise_disjoint():
    """Verify that server, validation, and client splits are mutually exclusive and balanced."""
    y = torch.tensor([c for c in range(10) for _ in range(5000)])  # 50k balanced targets
    C = 10
    seed = 42
    server_pc = 100
    val_pc = 200

    server, val, pool = make_splits(y, C, seed, server_pc, val_pc)

    # Check pairwise disjointness
    s_set, v_set, p_set = set(server), set(val), set(pool)
    assert len(s_set & v_set) == 0, "Server and validation splits overlap!"
    assert len(s_set & p_set) == 0, "Server and client pool splits overlap!"
    assert len(v_set & p_set) == 0, "Validation and client pool splits overlap!"

    # Check complete partition coverage
    assert len(s_set | v_set | p_set) == len(y)
    assert len(server) == C * server_pc
    assert len(val) == C * val_pc
    assert len(pool) == len(y) - (C * (server_pc + val_pc))


def test_prefix_merge_boundary_conditions():
    """Verify prefix merge boundaries: j=4 equals full merge, j=-1 equals full ensemble."""
    mA = ResNet20LN(num_classes=10)
    mB = ResNet20LN(num_classes=10)
    sdA = mA.state_dict()
    sdB = mB.state_dict()
    nb = len(mA.stages)

    x = torch.randn(8, 3, 32, 32)

    # Full merge baseline
    sdM = {k: 0.5 * (sdA[k].float() + sdB[k].float()) for k in sdA}
    mM = ResNet20LN(num_classes=10)
    mM.load_state_dict(sdM)
    mM.eval()
    pred_M = F.softmax(mM(x), dim=1)

    # Prefix j=4 (all stages merged)
    keys_all = {k for k in sdA if stage_of(k, nb) <= 4}
    assert keys_all == set(sdA.keys())

    hA = ResNet20LN(num_classes=10)
    hB = ResNet20LN(num_classes=10)
    hA.load_state_dict({k: (0.5 * (sdA[k].float() + sdB[k].float()) if k in keys_all else sdA[k]) for k in sdA})
    hB.load_state_dict({k: (0.5 * (sdA[k].float() + sdB[k].float()) if k in keys_all else sdB[k]) for k in sdA})
    hA.eval(); hB.eval()
    pred_j4 = 0.5 * (F.softmax(hA(x), dim=1) + F.softmax(hB(x), dim=1))

    # Boundary test 1: j=4 must match full merge exactly
    assert torch.allclose(pred_M, pred_j4, atol=1e-6)

    # Boundary test 2: j=-1 (no stages merged) must match full ensemble exactly
    mA.eval(); mB.eval()
    pred_ens = 0.5 * (F.softmax(mA(x), dim=1) + F.softmax(mB(x), dim=1))

    keys_none = {k for k in sdA if stage_of(k, nb) <= -1}
    assert len(keys_none) == 0

    hA_none = ResNet20LN(num_classes=10)
    hB_none = ResNet20LN(num_classes=10)
    hA_none.load_state_dict({k: (0.5 * (sdA[k].float() + sdB[k].float()) if k in keys_none else sdA[k]) for k in sdA})
    hB_none.load_state_dict({k: (0.5 * (sdA[k].float() + sdB[k].float()) if k in keys_none else sdB[k]) for k in sdA})
    hA_none.eval(); hB_none.eval()
    pred_j_none = 0.5 * (F.softmax(hA_none(x), dim=1) + F.softmax(hB_none(x), dim=1))

    assert torch.allclose(pred_ens, pred_j_none, atol=1e-6)
