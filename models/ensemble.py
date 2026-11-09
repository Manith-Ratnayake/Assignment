from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor

MODELS = {
    "random_forest": RandomForestRegressor,
    "extra_trees": ExtraTreesRegressor,
    "gradient_boosting": GradientBoostingRegressor,
}


def create_ensemble(name, params):
    if name in MODELS:
        return MODELS[name](**params)
    if name == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(**params)
    if name == "lightgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(**params)
    if name == "catboost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(**params)
    raise ValueError(f"Unknown ensemble model: {name}")