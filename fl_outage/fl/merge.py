from collections import OrderedDict
from copy import deepcopy
from typing import Optional
import torch
import torch.nn as nn

from fl_outage.fl.server import average_model_weights


def merge_models(
    model_A: nn.Module,
    model_B: nn.Module,
    rule: str = "plain_avg",
    anchor_model: Optional[nn.Module] = None,
    anchor_weight: float = 0.2
) -> nn.Module:
    """Reconciles two group models into a single global model.
    
    Supported rules:
    - 'plain_avg': 0.5 * (W_A + W_B)
    - 'anchored': anchor_weight * W_k + (1 - anchor_weight) * 0.5 * (W_A + W_B)
    - 'discard': W_k (discards outage training, resumes from anchor)
    """
    merged_model = deepcopy(model_A)

    if rule == "plain_avg":
        avg_state = average_model_weights([model_A.state_dict(), model_B.state_dict()])
        merged_model.load_state_dict(avg_state)
    elif rule == "anchored":
        assert anchor_model is not None, "anchor_model must be provided for anchored merging"
        avg_AB = average_model_weights([model_A.state_dict(), model_B.state_dict()])
        state_k = anchor_model.state_dict()
        anchored_state = OrderedDict()
        for k in avg_AB.keys():
            if torch.is_floating_point(avg_AB[k]):
                anchored_state[k] = anchor_weight * state_k[k] + (1.0 - anchor_weight) * avg_AB[k]
            else:
                anchored_state[k] = avg_AB[k]
        merged_model.load_state_dict(anchored_state)
    elif rule == "discard":
        assert anchor_model is not None, "anchor_model must be provided for discard rule"
        merged_model.load_state_dict(anchor_model.state_dict())
    else:
        raise ValueError(f"Unknown merge rule: {rule}")

    return merged_model
