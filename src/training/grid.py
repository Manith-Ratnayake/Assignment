import json

from sklearn.model_selection import ParameterGrid


def grid_combinations(path):
    """Read a grid file and yield (model_name, params, run_name) for every combination of values.

    The file looks like {"model_name": {"setting": [value, value, ...]}, ...}.
    An empty {} for a model means one run with its default settings.
    """
    with open(path) as file:
        grids = json.load(file)

    for model_name, grid in grids.items():
        for params in ParameterGrid(grid):
            run_name = "_".join(["grid", model_name] + [f"{key}{value}" for key, value in params.items()])
            yield model_name, params, run_name