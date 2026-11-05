from fl_outage.fl.client import FederatedClient
from fl_outage.fl.server import FederatedServer, FedAvgServer, average_model_weights
from fl_outage.fl.outage import OutageSimulator
from fl_outage.fl.merge import merge_models
from fl_outage.fl.recovery import run_recovery_evaluation

# Backward compatibility alias
FedAvgTrunk = FederatedServer

__all__ = [
    "FederatedClient",
    "FederatedServer",
    "FedAvgServer",
    "FedAvgTrunk",
    "average_model_weights",
    "OutageSimulator",
    "merge_models",
    "run_recovery_evaluation",
]
