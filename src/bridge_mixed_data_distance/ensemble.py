"""Generalized Bayesian graph-ensemble utilities for BRIDGE."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GraphEnsembleResult:
    """Output of generalized Bayesian graph averaging."""

    distance: np.ndarray
    pairwise_sd: np.ndarray
    weights: np.ndarray
    entropy: float
    effective_graphs: float


def gibbs_graph_weights(
    losses,
    *,
    eta: float = 8.0,
    edge_counts=None,
    edge_penalty: float = 0.08,
    prior=None,
) -> np.ndarray:
    """Compute generalized Bayesian Gibbs weights over graphs.

    Parameters
    ----------
    losses:
        Predictive-risk loss for each candidate graph.

    eta:
        Inverse-temperature parameter. Larger values concentrate
        posterior mass more strongly on low-loss graphs.

    edge_counts:
        Number of directed edges in each graph.

    edge_penalty:
        Complexity penalty applied per graph edge.

    prior:
        Optional prior probabilities over graphs. If omitted,
        a uniform prior is used.

    Returns
    -------
    numpy.ndarray
        Normalized Gibbs weights.
    """

    losses = np.asarray(losses, dtype=float)

    if losses.ndim != 1:
        raise ValueError("losses must be one-dimensional.")

    n_graphs = len(losses)

    if n_graphs == 0:
        raise ValueError("At least one graph is required.")

    if eta <= 0:
        raise ValueError("eta must be positive.")

    if edge_penalty < 0:
        raise ValueError("edge_penalty must be non-negative.")

    if edge_counts is None:
        edge_counts = np.zeros(n_graphs, dtype=float)
    else:
        edge_counts = np.asarray(edge_counts, dtype=float)

    if edge_counts.shape != losses.shape:
        raise ValueError(
            "edge_counts must have the same length as losses."
        )

    if prior is None:
        prior = np.full(
            n_graphs,
            1.0 / n_graphs,
            dtype=float,
        )
    else:
        prior = np.asarray(prior, dtype=float)

        if prior.shape != losses.shape:
            raise ValueError(
                "prior must have the same length as losses."
            )

        if np.any(prior < 0):
            raise ValueError(
                "prior probabilities cannot be negative."
            )

        total_prior = float(np.sum(prior))

        if total_prior <= 0:
            raise ValueError(
                "prior probabilities must sum to a positive value."
            )

        prior = prior / total_prior

    penalized_loss = (
        losses
        + edge_penalty * edge_counts
    )

    log_weights = (
        np.log(
            np.maximum(
                prior,
                np.finfo(float).tiny,
            )
        )
        - eta * penalized_loss
    )

    log_weights -= np.max(log_weights)

    weights = np.exp(log_weights)

    weights /= np.sum(weights)

    return weights


def graph_average_distance(
    distance_matrices,
    *,
    losses,
    eta: float = 8.0,
    edge_counts=None,
    edge_penalty: float = 0.08,
    prior=None,
) -> GraphEnsembleResult:
    """Average graph-specific BRIDGE distances using Gibbs weights.

    Parameters
    ----------
    distance_matrices:
        Sequence or array with shape
        ``(n_graphs, n_observations, n_observations)``.

    losses:
        Predictive-risk loss for every candidate graph.

    eta:
        Generalized Bayesian inverse temperature.

    edge_counts:
        Number of edges in each graph.

    edge_penalty:
        Complexity penalty per edge.

    prior:
        Optional prior probabilities over graphs.

    Returns
    -------
    GraphEnsembleResult
        Posterior mean distance, pair-specific posterior standard
        deviation, graph weights, entropy, and effective graph count.
    """

    distances = np.asarray(
        distance_matrices,
        dtype=float,
    )

    if distances.ndim != 3:
        raise ValueError(
            "distance_matrices must have shape "
            "(n_graphs, n, n)."
        )

    n_graphs, n_rows, n_cols = distances.shape

    if n_rows != n_cols:
        raise ValueError(
            "Each graph-specific distance matrix must be square."
        )

    if len(losses) != n_graphs:
        raise ValueError(
            "The number of losses must equal the number of graphs."
        )

    for matrix in distances:
        if not np.allclose(
            matrix,
            matrix.T,
            atol=1e-10,
        ):
            raise ValueError(
                "All graph-specific distance matrices must be symmetric."
            )

    weights = gibbs_graph_weights(
        losses,
        eta=eta,
        edge_counts=edge_counts,
        edge_penalty=edge_penalty,
        prior=prior,
    )

    posterior_mean = np.tensordot(
        weights,
        distances,
        axes=(0, 0),
    )

    centered = (
        distances
        - posterior_mean[None, :, :]
    )

    posterior_variance = np.tensordot(
        weights,
        centered ** 2,
        axes=(0, 0),
    )

    pairwise_sd = np.sqrt(
        np.maximum(
            posterior_variance,
            0.0,
        )
    )

    safe_weights = weights[
        weights > 0
    ]

    entropy = float(
        -np.sum(
            safe_weights
            * np.log(safe_weights)
        )
    )

    effective_graphs = float(
        np.exp(entropy)
    )

    np.fill_diagonal(
        posterior_mean,
        0.0,
    )

    np.fill_diagonal(
        pairwise_sd,
        0.0,
    )

    return GraphEnsembleResult(
        distance=posterior_mean,
        pairwise_sd=pairwise_sd,
        weights=weights,
        entropy=entropy,
        effective_graphs=effective_graphs,
    )