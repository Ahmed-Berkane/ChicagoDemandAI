from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import BOOL_COLS, CAT_COLS, NUM_COLS


def build_preprocessor(
    *,
    num_cols: list[str] | None = None,
    bool_cols: list[str] | None = None,
    cat_cols: list[str] | None = None,
) -> ColumnTransformer:
    num_cols = num_cols or NUM_COLS
    bool_cols = bool_cols or BOOL_COLS
    cat_cols = cat_cols or CAT_COLS
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    boolean_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipeline, num_cols),
        ("cat", categorical_pipeline, cat_cols),
        ("bool", boolean_pipeline, bool_cols),
    ])
