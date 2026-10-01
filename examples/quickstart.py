import numpy as np
import pandas as pd

from bridge_mixed_data_distance import bridge_distance


rng = np.random.default_rng(42)

n = 50

x1 = rng.normal(size=n)
x2 = rng.normal(size=n)

data = pd.DataFrame(
    {
        "x1": x1,
        "x2": x2,
        "z": x1 + 0.02 * rng.normal(size=n),
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
    random_state=42,
    return_details=True,
)

print("BRIDGE version test")
print("-------------------")

print("Distance matrix shape:")
print(result.distance.shape)

print("\nNovelty weights:")
print(result.novelty_weights)

print("\nSelected predictive models:")
print(result.selected_models)

print("\nBase variables:")
print(result.base_variables)

print("\nConditional variables:")
print(result.conditional_variables)

print("\nSymmetric matrix:")
print(np.allclose(result.distance, result.distance.T))

print("\nZero diagonal:")
print(np.allclose(np.diag(result.distance), 0.0))

print("\nFirst 5 x 5 distances:")
print(np.round(result.distance[:5, :5], 4))