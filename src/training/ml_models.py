from functools import partial

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

ML_MODELS = {
    "linear": LinearRegression,
    "ridge": Ridge,
    "lasso": Lasso,
    "elasticnet": ElasticNet,
    "decision_tree": DecisionTreeRegressor,
    "random_forest": RandomForestRegressor,
    "extra_trees": ExtraTreesRegressor,
    "gradient_boosting": GradientBoostingRegressor,
    "xgboost": XGBRegressor,
    "lightgbm": partial(LGBMRegressor, verbose=-1),     # silence per-iteration logs
    "catboost": partial(CatBoostRegressor, verbose=0),
}


def build_ml_model(name, params, seed):
    """Create the model called `name` with `params`, seeded so results are repeatable."""
    if name not in ML_MODELS:
        raise ValueError(f"Unknown ML model '{name}'. Choose from {list(ML_MODELS)}.")

    model = ML_MODELS[name](**params)

    # Seed models that use randomness (trees, boosting), unless params already set one
    current = model.get_params()
    if "random_state" in current and current["random_state"] is None:
        model.set_params(random_state=seed)
    return model
