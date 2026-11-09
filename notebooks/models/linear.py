from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge

MODELS = {
    "linear": LinearRegression,
    "ridge": Ridge,
    "lasso": Lasso,
    "elasticnet": ElasticNet,
}
