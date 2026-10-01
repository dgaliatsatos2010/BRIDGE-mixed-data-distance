"""Core implementation of BRIDGE.

BRIDGE:
Bayesian Redundancy-Aware Information Distance with
Graph Ensembles for Mixed-Type Data.

This module implements the redundancy-aware distance layer.
The graph-ensemble extension is implemented separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from scipy.spatial.distance import pdist, squareform

from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


_VALID_TYPES = {"num", "nom", "ord"}


@dataclass
class BridgeResult:
    """Detailed result returned by BRIDGE."""

    distance: np.ndarray
    novelty_weights: dict[str, float]
    selected_models: dict[str, str]
    base_variables: list[str]
    conditional_variables: list[str]


def _validate_kind(kind: str) -> str:
    """Normalize a variable-type label."""

    aliases = {
        "numeric": "num",
        "numerical": "num",
        "continuous": "num",
        "float": "num",
        "integer": "num",
        "categorical": "nom",
        "nominal": "nom",
        "category": "nom",
        "binary": "nom",
        "ordinal": "ord",
    }

    kind = str(kind).strip().lower()
    kind = aliases.get(kind, kind)

    if kind not in _VALID_TYPES:
        raise ValueError(
            f"Unknown variable type '{kind}'. "
            "Use 'num', 'nom', or 'ord'."
        )

    return kind


def _pairwise_absolute(x: Sequence) -> np.ndarray:
    """Pairwise absolute distances for one-dimensional values."""

    a = np.asarray(x, dtype=float).reshape(-1, 1)
    return squareform(pdist(a, metric="cityblock"))


def _pairwise_mismatch(x: Sequence) -> np.ndarray:
    """Pairwise mismatch distances for nominal values."""

    a = np.asarray(x)
    return (a[:, None] != a[None, :]).astype(float)


def _mean_off_diagonal(distance: np.ndarray) -> float:
    """Mean of the upper-triangular pairwise distances."""

    n = distance.shape[0]

    if n < 2:
        return 0.0

    indices = np.triu_indices(n, 1)
    return float(np.mean(distance[indices]))


def _component_distance(
    train_values: Sequence,
    eval_values: Sequence,
    kind: str,
) -> np.ndarray:
    """Construct a normalized univariate distance component."""

    kind = _validate_kind(kind)

    if kind in {"num", "ord"}:
        train_distance = _pairwise_absolute(train_values)
        eval_distance = _pairwise_absolute(eval_values)
    else:
        train_distance = _pairwise_mismatch(train_values)
        eval_distance = _pairwise_mismatch(eval_values)

    scale = _mean_off_diagonal(train_distance)

    if scale <= 1e-12:
        return np.zeros_like(eval_distance, dtype=float)

    return eval_distance / scale


def _encode_parents(
    train: pd.DataFrame,
    test: pd.DataFrame,
    parents: Sequence[str],
    variable_types: Mapping[str, str],
) -> tuple[np.ndarray, np.ndarray]:
    """Encode mixed-type parent variables for predictive models."""

    train_parts = []
    test_parts = []

    for column in parents:

        kind = _validate_kind(variable_types[column])

        if kind in {"num", "ord"}:

            tr = pd.to_numeric(train[column], errors="coerce")
            te = pd.to_numeric(test[column], errors="coerce")

            median = float(tr.median()) if tr.notna().any() else 0.0

            tr = tr.fillna(median).to_numpy(dtype=float).reshape(-1, 1)
            te = te.fillna(median).to_numpy(dtype=float).reshape(-1, 1)

            train_parts.append(tr)
            test_parts.append(te)

        else:

            levels = list(pd.Series(train[column]).dropna().unique())

            if not levels:
                levels = ["__missing__"]

            tr_columns = []
            te_columns = []

            for level in levels:
                tr_columns.append(
                    (train[column].to_numpy() == level)
                    .astype(float)
                    .reshape(-1, 1)
                )

                te_columns.append(
                    (test[column].to_numpy() == level)
                    .astype(float)
                    .reshape(-1, 1)
                )

            train_parts.append(np.hstack(tr_columns))
            test_parts.append(np.hstack(te_columns))

    if not train_parts:
        raise ValueError("At least one parent variable is required.")

    return np.hstack(train_parts), np.hstack(test_parts)


def _classifier_probabilities(
    model,
    X: np.ndarray,
    classes: np.ndarray,
) -> np.ndarray:
    """Align predicted probabilities to the global class order."""

    predicted = model.predict_proba(X)

    output = np.zeros((len(X), len(classes)), dtype=float)

    mapping = {
        class_value: index
        for index, class_value in enumerate(model.classes_)
    }

    for column_index, class_value in enumerate(classes):
        if class_value in mapping:
            output[:, column_index] = predicted[
                :, mapping[class_value]
            ]

    return output


def _numeric_predictive_envelope(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int,
    cv_splits: int,
):
    """Estimate predictive novelty for a numerical variable."""

    y = np.asarray(y, dtype=float)

    baseline_loss = float(np.mean((y - np.mean(y)) ** 2))

    if baseline_loss <= 1e-12:

        model = DummyRegressor(strategy="mean")
        model.fit(X, y)

        residual = np.zeros_like(y)

        return 0.0, model, residual, "constant"

    n_splits = min(cv_splits, len(y))

    if n_splits < 2:
        raise ValueError(
            "At least two observations are required."
        )

    cv = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    candidates = [
        (
            "linear",
            LinearRegression(),
        ),
        (
            "polynomial_degree_2",
            make_pipeline(
                PolynomialFeatures(
                    degree=2,
                    include_bias=False,
                ),
                LinearRegression(),
            ),
        ),
        (
            "decision_tree",
            DecisionTreeRegressor(
                max_depth=7,
                min_samples_leaf=2,
                random_state=random_state,
            ),
        ),
    ]

    best = None

    for name, estimator in candidates:

        predictions = np.zeros(len(y), dtype=float)

        for train_index, validation_index in cv.split(X):

            fitted = clone(estimator)

            fitted.fit(
                X[train_index],
                y[train_index],
            )

            predictions[validation_index] = fitted.predict(
                X[validation_index]
            )

        loss = float(
            np.mean(
                (y - predictions) ** 2
            )
        )

        if best is None or loss < best[0]:
            best = (
                loss,
                name,
                estimator,
                predictions,
            )

    loss, name, estimator, predictions = best

    q = float(
        np.clip(
            loss / baseline_loss,
            0.0,
            1.0,
        )
    )

    fitted_model = clone(estimator)
    fitted_model.fit(X, y)

    residual = y - predictions

    return q, fitted_model, residual, name


def _categorical_predictive_envelope(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int,
    cv_splits: int,
):
    """Estimate predictive novelty for a categorical variable."""

    y = np.asarray(y)

    classes, counts = np.unique(
        y,
        return_counts=True,
    )

    if len(classes) == 1:

        model = DummyClassifier(
            strategy="most_frequent"
        )

        model.fit(X, y)

        residual = np.zeros(
            (len(y), 1),
            dtype=float,
        )

        return (
            0.0,
            model,
            residual,
            "constant",
            classes,
        )

    Y = np.column_stack(
        [
            (y == class_value).astype(float)
            for class_value in classes
        ]
    )

    marginal = np.mean(Y, axis=0)

    baseline_loss = float(
        np.mean(
            np.sum(
                (Y - marginal) ** 2,
                axis=1,
            )
        )
    )

    minimum_class_size = int(
        np.min(counts)
    )

    n_splits = min(
        cv_splits,
        minimum_class_size,
    )

    if n_splits < 2:

        model = DummyClassifier(
            strategy="prior"
        )

        model.fit(X, y)

        probability = _classifier_probabilities(
            model,
            X,
            classes,
        )

        residual = Y - probability

        return (
            1.0,
            model,
            residual,
            "prior",
            classes,
        )

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    candidates = [
        (
            "logistic",
            LogisticRegression(
                max_iter=1000,
            ),
        ),
        (
            "polynomial_logistic",
            make_pipeline(
                PolynomialFeatures(
                    degree=2,
                    include_bias=False,
                ),
                LogisticRegression(
                    max_iter=1000,
                ),
            ),
        ),
        (
            "decision_tree",
            DecisionTreeClassifier(
                max_depth=5,
                min_samples_leaf=1,
                random_state=random_state,
            ),
        ),
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=100,
                min_samples_leaf=5,
                max_features=1.0,
                random_state=random_state,
                n_jobs=1,
            ),
        ),
    ]

    best = None

    for name, estimator in candidates:

        probabilities = np.zeros_like(
            Y,
            dtype=float,
        )

        for train_index, validation_index in cv.split(
            X,
            y,
        ):

            fitted = clone(estimator)

            fitted.fit(
                X[train_index],
                y[train_index],
            )

            probabilities[
                validation_index
            ] = _classifier_probabilities(
                fitted,
                X[validation_index],
                classes,
            )

        loss = float(
            np.mean(
                np.sum(
                    (Y - probabilities) ** 2,
                    axis=1,
                )
            )
        )

        if best is None or loss < best[0]:

            best = (
                loss,
                name,
                estimator,
                probabilities,
            )

    loss, name, estimator, probabilities = best

    if baseline_loss <= 1e-12:
        q = 0.0
    else:
        q = float(
            np.clip(
                loss / baseline_loss,
                0.0,
                1.0,
            )
        )

    fitted_model = clone(estimator)
    fitted_model.fit(X, y)

    residual = Y - probabilities

    return (
        q,
        fitted_model,
        residual,
        name,
        classes,
    )


def bridge_distance(
    train: pd.DataFrame,
    test: pd.DataFrame | None = None,
    *,
    variable_types: Mapping[str, str],
    conditional_parents: Mapping[
        str,
        Sequence[str],
    ] | None = None,
    random_state: int = 0,
    cv_splits: int = 3,
    return_details: bool = False,
) -> np.ndarray | BridgeResult:
    """Compute the BRIDGE pairwise distance matrix.

    Parameters
    ----------
    train:
        Training/reference data used to estimate normalization
        constants and predictive novelty.

    test:
        Observations for which pairwise distances are required.
        If omitted, distances are computed within ``train``.

    variable_types:
        Mapping from variable name to variable type.
        Accepted types are ``"num"``, ``"nom"``, and ``"ord"``.

    conditional_parents:
        Mapping of a potentially redundant variable to the variables
        used to predict it.

        Example::

            {
                "z": ["x1"],
                "xor_feature": ["b1", "b2"],
            }

    random_state:
        Random seed used during predictive cross-validation.

    cv_splits:
        Maximum number of cross-validation folds.

    return_details:
        When True, return a :class:`BridgeResult` containing the
        distance matrix, novelty weights, and selected predictive
        models.

    Returns
    -------
    numpy.ndarray or BridgeResult
        BRIDGE pairwise-distance matrix, optionally with diagnostics.
    """

    if not isinstance(train, pd.DataFrame):
        raise TypeError(
            "train must be a pandas DataFrame."
        )

    if test is None:
        test = train.copy()

    if not isinstance(test, pd.DataFrame):
        raise TypeError(
            "test must be a pandas DataFrame."
        )

    conditional_parents = (
        {}
        if conditional_parents is None
        else dict(conditional_parents)
    )

    variable_types = {
        column: _validate_kind(kind)
        for column, kind in variable_types.items()
    }

    required_variables = set(
        variable_types
    )

    missing_train = required_variables.difference(
        train.columns
    )

    missing_test = required_variables.difference(
        test.columns
    )

    if missing_train:
        raise ValueError(
            f"Variables missing from train: "
            f"{sorted(missing_train)}"
        )

    if missing_test:
        raise ValueError(
            f"Variables missing from test: "
            f"{sorted(missing_test)}"
        )

    conditional_variables = set(
        conditional_parents
    )

    for child, parents in conditional_parents.items():

        if child not in variable_types:
            raise ValueError(
                f"Unknown conditional variable: {child}"
            )

        for parent in parents:
            if parent not in variable_types:
                raise ValueError(
                    f"Unknown parent variable: {parent}"
                )

    base_variables = [
        column
        for column in variable_types
        if column not in conditional_variables
    ]

    if not base_variables:
        raise ValueError(
            "BRIDGE requires at least one base variable."
        )

    base_components = []

    for column in base_variables:

        component = _component_distance(
            train[column],
            test[column],
            variable_types[column],
        )

        base_components.append(component)

    numerator = np.sum(
        base_components,
        axis=0,
    )

    denominator = float(
        len(base_components)
    )

    novelty_weights = {}
    selected_models = {}

    for index, (
        child,
        parents,
    ) in enumerate(
        conditional_parents.items()
    ):

        kind = variable_types[child]

        X_train, X_test = _encode_parents(
            train,
            test,
            parents,
            variable_types,
        )

        seed = random_state + index

        if kind == "num":

            y_train = pd.to_numeric(
                train[child],
                errors="raise",
            ).to_numpy(dtype=float)

            y_test = pd.to_numeric(
                test[child],
                errors="raise",
            ).to_numpy(dtype=float)

            (
                q,
                fitted_model,
                training_residual,
                model_name,
            ) = _numeric_predictive_envelope(
                X_train,
                y_train,
                seed,
                cv_splits,
            )

            test_residual = (
                y_test
                - fitted_model.predict(X_test)
            )

            training_residual_distance = (
                _pairwise_absolute(
                    training_residual
                )
            )

            test_residual_distance = (
                _pairwise_absolute(
                    test_residual
                )
            )

        else:

            y_train = train[
                child
            ].to_numpy()

            y_test = test[
                child
            ].to_numpy()

            (
                q,
                fitted_model,
                training_residual,
                model_name,
                classes,
            ) = _categorical_predictive_envelope(
                X_train,
                y_train,
                seed,
                cv_splits,
            )

            unseen = set(
                np.unique(y_test)
            ).difference(
                set(classes)
            )

            if unseen:
                raise ValueError(
                    f"Variable '{child}' contains "
                    f"unseen test categories: "
                    f"{sorted(unseen)}"
                )

            actual_test = np.column_stack(
                [
                    (
                        y_test
                        == class_value
                    ).astype(float)
                    for class_value in classes
                ]
            )

            predicted_test = (
                _classifier_probabilities(
                    fitted_model,
                    X_test,
                    classes,
                )
            )

            test_residual = (
                actual_test
                - predicted_test
            )

            training_residual_distance = (
                squareform(
                    pdist(
                        training_residual,
                        metric="cityblock",
                    )
                )
            )

            test_residual_distance = (
                squareform(
                    pdist(
                        test_residual,
                        metric="cityblock",
                    )
                )
            )

        residual_scale = (
            _mean_off_diagonal(
                training_residual_distance
            )
        )

        if residual_scale <= 1e-12:
            novelty_component = np.zeros_like(
                test_residual_distance,
                dtype=float,
            )
        else:
            novelty_component = (
                test_residual_distance
                / residual_scale
            )

        numerator = (
            numerator
            + q * novelty_component
        )

        denominator += q

        novelty_weights[child] = q
        selected_models[child] = model_name

    distance = (
        numerator / denominator
    )

    np.fill_diagonal(
        distance,
        0.0,
    )

    distance = (
        distance + distance.T
    ) / 2.0

    if return_details:

        return BridgeResult(
            distance=distance,
            novelty_weights=novelty_weights,
            selected_models=selected_models,
            base_variables=base_variables,
            conditional_variables=list(
                conditional_parents
            ),
        )

    return distance