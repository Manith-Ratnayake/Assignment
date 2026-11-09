from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

MODELS = {
    "Random Forest": RandomForestRegressor,
    "Extra Trees": ExtraTreesRegressor,
    "Gradient Boosting": GradientBoostingRegressor,
    "XGBoost": XGBRegressor,
    "LightGBM": LGBMRegressor,
    "CatBoost": CatBoostRegressor
}