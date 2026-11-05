"""Backward-compatibility module forwarding to fl_outage.fl.server."""

from fl_outage.fl.server import FederatedServer as FedAvgTrunk, average_model_weights, FederatedServer, FedAvgServer

__all__ = ["FedAvgTrunk", "FederatedServer", "FedAvgServer", "average_model_weights"]
