"""BRIDGE: Bayesian Redundancy-Aware Information Distance."""

from .core import (
    BridgeResult,
    bridge_distance,
)

from .ensemble import (
    GraphEnsembleResult,
    gibbs_graph_weights,
    graph_average_distance,
)

__version__ = "1.0.0"

__all__ = [
    "BridgeResult",
    "bridge_distance",
    "GraphEnsembleResult",
    "gibbs_graph_weights",
    "graph_average_distance",
]