from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    Normalizer
)

def get_scaler(name):
    return {
        "minmax": MinMaxScaler(),
        "standard": StandardScaler(),
        "robust": RobustScaler(),
        "l2": Normalizer(norm="l2")
    }.get(name)
