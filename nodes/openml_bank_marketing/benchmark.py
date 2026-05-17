from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, StandardScaler


NODE_NAME = "openml_bank_marketing"
DATA_ID = 1461
DATASET_NAME = "bank-marketing"
TARGET_NAME = "y"
SEED = 42
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15
CACHE_DIR = Path(__file__).resolve().parent / "data_cache"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"

ALLOWED_VALUES = {
    "model_type": {"logistic_regression", "random_forest", "gradient_boosting"},
    "class_weight": {None, "balanced"},
    "scaler": {"none", "standard", "minmax"},
    "imputer": {"most_frequent", "constant"},
}
ALLOWED_RANGES = {
    "C": (0.001, 100.0),
    "max_depth": (2, 30),
    "n_estimators": (20, 500),
    "learning_rate": (0.001, 0.5),
    "max_iter": (100, 3000),
}
REQUIRED_KEYS = (
    "model_type",
    "C",
    "max_depth",
    "n_estimators",
    "learning_rate",
    "class_weight",
    "scaler",
    "imputer",
    "max_iter",
)


class ConfigError(ValueError):
    """Raised when the editable node config violates the bounded schema."""


def run_openml_benchmark(config_path: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    x, y = _load_frame()
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y.astype(str))
    if len(encoder.classes_) != 2:
        raise RuntimeError(f"{DATASET_NAME} expected binary target, got {list(encoder.classes_)}")

    x_train, x_temp, y_train, y_temp = train_test_split(
        x,
        y_encoded,
        train_size=TRAIN_SIZE,
        random_state=SEED,
        stratify=y_encoded,
    )
    val_fraction_of_temp = VAL_SIZE / (VAL_SIZE + TEST_SIZE)
    x_val, x_test, y_val, _y_test = train_test_split(
        x_temp,
        y_temp,
        train_size=val_fraction_of_temp,
        random_state=SEED,
        stratify=y_temp,
    )

    pipeline = _build_pipeline(x_train, config)
    pipeline.fit(x_train, y_train)
    val_probability = pipeline.predict_proba(x_val)[:, 1]
    val_prediction = (val_probability >= 0.5).astype(int)
    val_auc = float(roc_auc_score(y_val, val_probability))
    val_accuracy = float(accuracy_score(y_val, val_prediction))
    return {
        "node": NODE_NAME,
        "dataset_id": DATA_ID,
        "dataset_name": DATASET_NAME,
        "target": TARGET_NAME,
        "seed": SEED,
        "split": {"train": TRAIN_SIZE, "val": VAL_SIZE, "test": TEST_SIZE},
        "n_train": int(len(x_train)),
        "n_val": int(len(x_val)),
        "n_test": int(len(x_test)),
        "positive_label": str(encoder.classes_[1]),
        "model_type": config["model_type"],
        "val_auc": val_auc,
        "val_accuracy": val_accuracy,
        "status": "success",
        "config": config,
    }


def emit_success(metrics: dict[str, Any]) -> None:
    lines = [
        f"NODE={metrics['node']}",
        f"MODEL={metrics['model_type']}",
        f"VAL_AUC={metrics['val_auc']:.6f}",
        f"VAL_ACC={metrics['val_accuracy']:.6f}",
        "STATUS=success",
    ]
    _write_artifacts(metrics, lines)
    print("\n".join(lines))


def emit_failure(category: str, message: str, config_path: Path) -> None:
    payload = {
        "node": NODE_NAME,
        "dataset_id": DATA_ID,
        "dataset_name": DATASET_NAME,
        "target": TARGET_NAME,
        "seed": SEED,
        "status": "failed",
        "failure_category": category,
        "failure_message": message,
        "config_path": str(config_path),
    }
    lines = [
        f"NODE={NODE_NAME}",
        "STATUS=failed",
        f"FAILURE_CATEGORY={category}",
        f"FAILURE_MESSAGE={message}",
    ]
    _write_artifacts(payload, lines)
    print("\n".join(lines))


def _load_config(config_path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError("config root must be a mapping")
    missing = [key for key in REQUIRED_KEYS if key not in raw]
    if missing:
        raise ConfigError(f"missing required config keys: {', '.join(missing)}")
    config = dict(raw)
    for key, allowed in ALLOWED_VALUES.items():
        if config[key] not in allowed:
            raise ConfigError(f"{key}={config[key]!r} is not in allowed values {sorted(str(v) for v in allowed)}")
    for key, (low, high) in ALLOWED_RANGES.items():
        value = float(config[key])
        if value < low or value > high:
            raise ConfigError(f"{key}={value} outside allowed range [{low}, {high}]")
    config["C"] = float(config["C"])
    config["max_depth"] = int(config["max_depth"])
    config["n_estimators"] = int(config["n_estimators"])
    config["learning_rate"] = float(config["learning_rate"])
    config["max_iter"] = int(config["max_iter"])
    return config


def _load_frame():
    bunch = fetch_openml(data_id=DATA_ID, as_frame=True, parser="auto", data_home=str(CACHE_DIR))
    if bunch.data is None or bunch.target is None:
        raise RuntimeError(f"OpenML dataset {DATA_ID} did not return data and target")
    return bunch.data.copy(), bunch.target.copy()


def _build_pipeline(x_train, config: dict[str, Any]) -> Pipeline:
    categorical_columns = [
        column for column in x_train.columns if str(x_train[column].dtype) in {"category", "object", "bool"}
    ]
    numeric_columns = [column for column in x_train.columns if column not in categorical_columns]
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if config["scaler"] == "standard":
        numeric_steps.append(("scaler", StandardScaler()))
    elif config["scaler"] == "minmax":
        numeric_steps.append(("scaler", MinMaxScaler()))
    categorical_imputer = (
        SimpleImputer(strategy="constant", fill_value="__missing__")
        if config["imputer"] == "constant"
        else SimpleImputer(strategy="most_frequent")
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", categorical_imputer),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=config["model_type"] != "gradient_boosting",
                ),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=numeric_steps), numeric_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("classifier", _build_model(config)),
        ]
    )


def _build_model(config: dict[str, Any]):
    model_type = config["model_type"]
    if model_type == "logistic_regression":
        return LogisticRegression(
            C=config["C"],
            class_weight=config["class_weight"],
            max_iter=config["max_iter"],
            solver="liblinear",
            random_state=SEED,
        )
    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            class_weight=config["class_weight"],
            random_state=SEED,
            n_jobs=1,
        )
    if model_type == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=config["n_estimators"],
            learning_rate=config["learning_rate"],
            max_depth=config["max_depth"],
            random_state=SEED,
        )
    raise ConfigError(f"unsupported model_type={model_type!r}")


def _write_artifacts(payload: dict[str, Any], lines: list[str]) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (ARTIFACTS_DIR / "run.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
