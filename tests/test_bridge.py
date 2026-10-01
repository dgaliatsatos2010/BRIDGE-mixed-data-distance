import numpy as np
import pandas as pd

from bridge_mixed_data_distance import (
    bridge_distance,
    gibbs_graph_weights,
    graph_average_distance,
)


def test_bridge_distance_properties():
    rng = np.random.default_rng(123)

    n = 40

    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)

    data = pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "z": x1 + 0.01 * rng.normal(size=n),
        }
    )

    result = bridge_distance(
        data,
        variable_types={
            "x1": "num",
            "x2": "num",
            "z": "num",
        },
        conditional_parents={
            "z": ["x1"],
        },
        random_state=123,
        return_details=True,
    )

    assert result.distance.shape == (n, n)

    assert np.allclose(
        result.distance,
        result.distance.T,
    )

    assert np.allclose(
        np.diag(result.distance),
        0.0,
    )

    assert np.all(
        result.distance >= 0
    )

    assert "z" in result.novelty_weights

    assert 0.0 <= result.novelty_weights["z"] <= 1.0


def test_exact_redundancy_has_low_novelty():
    rng = np.random.default_rng(321)

    n = 60

    x = rng.normal(size=n)

    data = pd.DataFrame(
        {
            "x": x,
            "duplicate": x.copy(),
        }
    )

    result = bridge_distance(
        data,
        variable_types={
            "x": "num",
            "duplicate": "num",
        },
        conditional_parents={
            "duplicate": ["x"],
        },
        random_state=321,
        return_details=True,
    )

    assert result.novelty_weights["duplicate"] < 0.05


def test_gibbs_weights_sum_to_one():
    losses = np.array(
        [
            0.10,
            0.20,
            0.30,
        ]
    )

    weights = gibbs_graph_weights(
        losses,
        eta=8.0,
        edge_counts=[
            1,
            1,
            2,
        ],
        edge_penalty=0.08,
    )

    assert np.isclose(
        weights.sum(),
        1.0,
    )

    assert np.all(
        weights >= 0
    )

    assert weights[0] > weights[1]


def test_graph_average_distance():
    d1 = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.5],
            [2.0, 1.5, 0.0],
        ]
    )

    d2 = np.array(
        [
            [0.0, 1.2, 1.8],
            [1.2, 0.0, 1.4],
            [1.8, 1.4, 0.0],
        ]
    )

    result = graph_average_distance(
        [
            d1,
            d2,
        ],
        losses=[
            0.10,
            0.20,
        ],
        eta=8.0,
        edge_counts=[
            1,
            1,
        ],
        edge_penalty=0.08,
    )

    assert result.distance.shape == (3, 3)

    assert result.pairwise_sd.shape == (3, 3)

    assert np.allclose(
        result.distance,
        result.distance.T,
    )

    assert np.allclose(
        np.diag(result.distance),
        0.0,
    )

    assert np.allclose(
        np.diag(result.pairwise_sd),
        0.0,
    )

    assert np.isclose(
        result.weights.sum(),
        1.0,
    )

    assert result.effective_graphs >= 1.0