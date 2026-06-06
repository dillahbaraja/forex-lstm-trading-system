import argparse
import hashlib
import json
import os
import shutil
import sys
import datetime
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
import tf2onnx
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Attention,
    Concatenate,
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    LSTM,
    MaxPooling1D,
)
from tensorflow.keras.models import Model, load_model

try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    import onnx
except Exception:
    convert_sklearn = None
    FloatTensorType = None
    onnx = None

import data_preprocessing as dp

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

SEQ_LENGTH = 24
NUM_CLASSES = 3
TRAINING_SCHEMA_VERSION = 3
MODEL_FAMILY = "single_head_binary_booster_v1"
BASE_FINAL_EPOCHS = 300
BASE_SWEEP_EPOCHS = 80
DEFAULT_CONFIDENCE_THRESHOLD = 0.55
DEFAULT_MIN_CONFIDENCE_EDGE = 0.08
MIN_COVERAGE_RATIO = 0.15
BOOSTER_MIN_COVERAGE_RATIO = 0.03
MIN_TRADE_COUNT_FLOOR_RATIO = 0.01
MIN_TRADE_COUNT_FLOOR = 20
MIN_CLASS_SUPPORT_RATIO = 0.01
ADAPTIVE_PATIENCE = 60
ADAPTIVE_MIN_EPOCHS = 40
ADAPTIVE_LR_PATIENCE = 15
RUNS_DIR_NAME = "training_runs"
RUN_INDEX_NAME = "index.jsonl"
LATEST_POINTER_NAME = "latest_run.json"
BEST_REGISTRY_NAME = "best_registry.json"
DEFAULT_SYMBOL = "EURUSD"
DEFAULT_TIMEFRAME = "H1"
BOOSTER_SKIP_CLASS = 0
BOOSTER_ENTER_CLASS = 1
BOOSTER_INPUT_FEATURE_COUNT = 30
AUX_HORIZONS = ()
BOOSTER_CANDIDATES = [
    {"n_estimators": 90, "learning_rate": 0.05, "max_depth": 2, "min_samples_leaf": 18, "subsample": 0.85},
    {"n_estimators": 130, "learning_rate": 0.04, "max_depth": 2, "min_samples_leaf": 14, "subsample": 0.85},
    {"n_estimators": 170, "learning_rate": 0.03, "max_depth": 3, "min_samples_leaf": 20, "subsample": 0.80},
]

LABEL_CANDIDATES = [
    {"future_horizon": 18, "tp_mult": 1.8, "sl_mult": 1.2, "min_move_atr": 0.35, "neutral_margin_atr": 0.05},
    {"future_horizon": 24, "tp_mult": 2.0, "sl_mult": 1.5, "min_move_atr": 0.50, "neutral_margin_atr": 0.08},
    {"future_horizon": 24, "tp_mult": 2.2, "sl_mult": 1.5, "min_move_atr": 0.65, "neutral_margin_atr": 0.10},
    {"future_horizon": 30, "tp_mult": 2.4, "sl_mult": 1.6, "min_move_atr": 0.75, "neutral_margin_atr": 0.12},
]


@dataclass
class TrainPaths:
    project_root: str
    symbol: str
    timeframe: str
    pair_slug: str
    history_dir: str
    run_id: str
    run_dir: str
    checkpoint_dir: str
    best_path: str
    last_path: str
    state_path: str
    index_path: str
    latest_pointer_path: str
    best_registry_path: str
    onnx_path: str
    booster_onnx_path: str
    report_path: str
    latest_report_path: str
    latest_file_report_path: str
    feature_contract_path: str
    label_config_path: str
    scaler_path: str
    scaler_file_path: str
    feature_contract_file_path: str
    label_config_file_path: str


def _json_hash(payload: dict) -> str:
    normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _feature_contract(feature_columns):
    return {
        "feature_count": len(feature_columns),
        "sequence_length": SEQ_LENGTH,
        "feature_columns": list(feature_columns),
    }


def _feature_contract_hash(feature_columns):
    return _json_hash(_feature_contract(feature_columns))


def _normalize_symbol(symbol: str | None) -> str:
    return (symbol or DEFAULT_SYMBOL).strip().upper()


def _normalize_timeframe(timeframe: str | None) -> str:
    return (timeframe or DEFAULT_TIMEFRAME).strip().upper()


def _pair_slug(symbol: str, timeframe: str) -> str:
    return f"{symbol.lower()}_{timeframe}"


def _artifact_name(prefix: str, suffix: str, ext: str) -> str:
    return f"{prefix}_{suffix}.{ext}"


def _is_legacy_default_pair(symbol: str, timeframe: str) -> bool:
    return _normalize_symbol(symbol) == DEFAULT_SYMBOL and _normalize_timeframe(timeframe) == DEFAULT_TIMEFRAME


def _default_csv_name(symbol: str, timeframe: str) -> str:
    return f"{_normalize_symbol(symbol)}_{_normalize_timeframe(timeframe)}_Data.csv"


def _parse_args():
    parser = argparse.ArgumentParser(description="Train the LSTM trading model for a specific symbol and timeframe.")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL, help="Trading symbol, for example EURUSD, USDJPY, or EURJPY.")
    parser.add_argument("--timeframe", default=DEFAULT_TIMEFRAME, help="Timeframe label used in file naming. Default: H1.")
    parser.add_argument("--csv", default=None, help="Optional explicit CSV path. Defaults to <SYMBOL>_<TIMEFRAME>_Data.csv in project root.")
    parser.add_argument(
        "--strict-eurusd-baseline",
        action="store_true",
        help="Force non-EURUSD pairs to reuse the final EURUSD label configuration for strict apple-to-apple comparison.",
    )
    return parser.parse_args()


def _load_strict_eurusd_label_params(project_root: str) -> dict:
    report_path = os.path.join(project_root, "training_report.json")
    if not os.path.exists(report_path):
        raise FileNotFoundError("training_report.json not found. Strict EURUSD baseline mode requires the final EURUSD report.")

    with open(report_path, "r", encoding="utf-8") as handle:
        report = json.load(handle)

    label_params = report.get("selected_label_params")
    if not label_params:
        raise ValueError("training_report.json does not contain selected_label_params required for strict EURUSD baseline mode.")

    return dict(label_params)


def _paths_for(project_root: str, run_id: str, feature_contract_hash: str, label_hash: str, symbol: str, timeframe: str) -> TrainPaths:
    history_dir = os.path.join(project_root, RUNS_DIR_NAME)
    run_dir = os.path.join(history_dir, run_id)
    pair_slug = _pair_slug(symbol, timeframe)
    return TrainPaths(
        project_root=project_root,
        symbol=symbol,
        timeframe=timeframe,
        pair_slug=pair_slug,
        history_dir=history_dir,
        run_id=run_id,
        run_dir=run_dir,
        checkpoint_dir=run_dir,
        best_path=os.path.join(run_dir, "best.keras"),
        last_path=os.path.join(run_dir, "last.keras"),
        state_path=os.path.join(run_dir, "training_state.json"),
        index_path=os.path.join(history_dir, _artifact_name("index", pair_slug, "jsonl")),
        latest_pointer_path=os.path.join(history_dir, _artifact_name("latest_run", pair_slug, "json")),
        best_registry_path=os.path.join(history_dir, _artifact_name("best_registry", pair_slug, "json")),
        onnx_path=os.path.join(project_root, _artifact_name("model", pair_slug, "onnx")),
        booster_onnx_path=os.path.join(project_root, _artifact_name("booster", pair_slug, "onnx")),
        report_path=os.path.join(run_dir, "run_report.json"),
        latest_report_path=os.path.join(project_root, _artifact_name("training_report", pair_slug, "json")),
        latest_file_report_path=os.path.join(project_root, "MQL5", "Files", _artifact_name("training_report", pair_slug, "json")),
        feature_contract_path=os.path.join(run_dir, "feature_contract.json"),
        label_config_path=os.path.join(run_dir, "label_config.json"),
        scaler_path=os.path.join(run_dir, "scaler_params.csv"),
        scaler_file_path=os.path.join(project_root, "MQL5", "Files", _artifact_name("scaler_params", pair_slug, "csv")),
        feature_contract_file_path=os.path.join(project_root, "MQL5", "Files", _artifact_name("feature_contract", pair_slug, "json")),
        label_config_file_path=os.path.join(project_root, "MQL5", "Files", _artifact_name("label_config", pair_slug, "json")),
    )


def _ensure_parent(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _save_json(path: str, payload: dict):
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _load_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _append_jsonl(path: str, payload: dict):
    _ensure_parent(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def _load_jsonl(path: str):
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _now_run_id(feature_contract_hash: str, label_config_hash: str, symbol: str, timeframe: str) -> str:
    stamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{_pair_slug(symbol, timeframe)}_seq{SEQ_LENGTH}_{feature_contract_hash[:8]}_{label_config_hash[:8]}"


def _history_paths(project_root: str):
    history_dir = os.path.join(project_root, RUNS_DIR_NAME)
    return {
        "history_dir": history_dir,
        "index_path": os.path.join(history_dir, RUN_INDEX_NAME),
        "latest_pointer_path": os.path.join(history_dir, LATEST_POINTER_NAME),
        "best_registry_path": os.path.join(history_dir, BEST_REGISTRY_NAME),
    }


def _score_tuple_from_metrics(metrics: dict):
    trading = metrics["trading_proxy"]
    return (
        float(trading["expectancy_r"]),
        float(trading["profit_factor"]),
        -float(trading["max_drawdown_r"]),
        float(metrics["macro_f1"]),
        float(metrics["weighted_f1"]),
    )


def _comparison_block(current_report: dict, previous_record: dict | None):
    if not previous_record:
        return {
            "previous_run_id": None,
            "new_best_model": True,
            "metric_deltas": {},
        }

    previous_validation = previous_record.get("best_validation_metrics") or previous_record.get("validation_metrics") or {}
    previous_test = previous_record.get("test_metrics") or {}
    current_validation = current_report.get("best_validation_metrics") or current_report.get("validation_metrics") or {}
    current_test = current_report.get("test_metrics") or {}

    def delta(cur, prev, path):
        node_cur = cur
        node_prev = prev
        for key in path:
            node_cur = node_cur[key]
            node_prev = node_prev[key]
        return float(node_cur) - float(node_prev)

    deltas = {}
    for key in ("accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall", "confidence_ge_threshold_ratio"):
        if key in current_validation and key in previous_validation:
            deltas[f"validation_{key}_delta"] = float(current_validation[key]) - float(previous_validation[key])
        if key in current_test and key in previous_test:
            deltas[f"test_{key}_delta"] = float(current_test[key]) - float(previous_test[key])

    for key in ("expectancy_r", "profit_factor", "max_drawdown_r", "trade_count", "win_rate"):
        if key in current_validation.get("trading_proxy", {}) and key in previous_validation.get("trading_proxy", {}):
            deltas[f"validation_trading_{key}_delta"] = float(current_validation["trading_proxy"][key]) - float(previous_validation["trading_proxy"][key])
        if key in current_test.get("trading_proxy", {}) and key in previous_test.get("trading_proxy", {}):
            deltas[f"test_trading_{key}_delta"] = float(current_test["trading_proxy"][key]) - float(previous_test["trading_proxy"][key])

    current_score = current_report.get("best_validation_score_tuple") or []
    previous_score = previous_record.get("best_validation_score_tuple") or []
    new_best_model = tuple(current_score) > tuple(previous_score)

    return {
        "previous_run_id": previous_record.get("run_id"),
        "previous_run_dir": previous_record.get("run_dir"),
        "previous_best_score_tuple": previous_score,
        "current_best_score_tuple": current_score,
        "new_best_model": bool(new_best_model),
        "metric_deltas": deltas,
    }


def _select_best_compatible_history(records, expected_state: dict):
    compatible = []
    for record in records:
        if record.get("feature_contract_hash") != expected_state.get("feature_contract_hash"):
            continue
        if int(record.get("sequence_length", -1)) != int(expected_state.get("sequence_length", -1)):
            continue
        if int(record.get("training_schema_version", -1)) != int(expected_state.get("training_schema_version", -1)):
            continue
        if record.get("model_family") != expected_state.get("model_family"):
            continue
        checkpoint_path = record.get("best_checkpoint_path")
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            continue
        score = record.get("best_validation_score_tuple")
        if score is None:
            continue
        compatible.append((tuple(score), record))

    if not compatible:
        return None

    compatible.sort(key=lambda item: item[0], reverse=True)
    return compatible[0][1]


def _load_resume_checkpoint_from_history(records, expected_state: dict):
    candidate = _select_best_compatible_history(records, expected_state)
    if not candidate:
        return None, 0, "cold_start", None, None

    checkpoint_path = candidate.get("best_checkpoint_path")
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return None, 0, "cold_start", None, candidate

    try:
        model = load_model(checkpoint_path)
        initial_epoch = int(candidate.get("epoch_completed", 0))
        return model, initial_epoch, "history_best_checkpoint", checkpoint_path, candidate
    except Exception as exc:
        print(f"Could not resume from historical checkpoint {checkpoint_path}: {exc}", flush=True)
        return None, 0, "cold_start", None, candidate


def _update_best_registry(best_registry_path: str, current_record: dict):
    registry = _load_json(best_registry_path) or {}
    current_score = tuple(current_record.get("best_validation_score_tuple") or [])
    existing = registry.get("overall_best")
    if existing:
        existing_score = tuple(existing.get("best_validation_score_tuple") or [])
        if current_score <= existing_score:
            registry["last_run"] = current_record
            _save_json(best_registry_path, registry)
            return registry

    registry["overall_best"] = {
        "run_id": current_record.get("run_id"),
        "run_dir": current_record.get("run_dir"),
        "best_checkpoint_path": current_record.get("best_checkpoint_path"),
        "feature_contract_hash": current_record.get("feature_contract_hash"),
        "label_config_hash": current_record.get("label_config_hash"),
        "sequence_length": current_record.get("sequence_length"),
        "training_schema_version": current_record.get("training_schema_version"),
        "model_family": current_record.get("model_family"),
        "best_validation_score_tuple": current_record.get("best_validation_score_tuple"),
        "best_validation_metrics": current_record.get("best_validation_metrics"),
        "test_metrics": current_record.get("test_metrics"),
    }
    registry["last_run"] = current_record
    _save_json(best_registry_path, registry)
    return registry


def _class_histogram(y):
    counts = np.bincount(np.asarray(y, dtype=int), minlength=NUM_CLASSES)
    return {str(i): int(v) for i, v in enumerate(counts.tolist())}


def _compute_class_weights(y_train):
    unique = np.unique(y_train)
    if len(unique) < 2:
        return {int(unique[0]): 1.0}

    class_weights_values = compute_class_weight(
        class_weight="balanced",
        classes=unique,
        y=y_train,
    )
    return {int(cls): float(weight) for cls, weight in zip(unique, class_weights_values)}


def _sequence_masks(end_indices, train_end, val_end):
    train_mask = end_indices < train_end
    val_mask = (end_indices >= train_end) & (end_indices < val_end)
    test_mask = end_indices >= val_end
    return train_mask, val_mask, test_mask


def _build_multitask_sequence_payload(scaled_features, main_targets, aux_targets_map, seq_length=SEQ_LENGTH):
    X, y_main, end_indices = dp.create_sequences(scaled_features, main_targets, seq_length=seq_length, return_end_indices=True)
    aux_sequences = {}
    for horizon, targets in aux_targets_map.items():
        _, y_aux, aux_end_indices = dp.create_sequences(scaled_features, targets, seq_length=seq_length, return_end_indices=True)
        if not np.array_equal(end_indices, aux_end_indices):
            raise ValueError(f"Auxiliary target sequence misalignment detected for horizon {horizon}.")
        aux_sequences[int(horizon)] = y_aux
    return {
        "X": X,
        "y_main": y_main,
        "aux_sequences": aux_sequences,
        "end_indices": end_indices,
    }


def _main_output_name(model) -> str:
    if isinstance(getattr(model, "output_names", None), list) and "main_output" in model.output_names:
        return "main_output"
    return "main_output"


def _predict_main_prob(model, X):
    predictions = model.predict(X, verbose=0)
    if isinstance(predictions, dict):
        if "main_output" in predictions:
            return np.asarray(predictions["main_output"], dtype=np.float32)
        first_key = next(iter(predictions))
        return np.asarray(predictions[first_key], dtype=np.float32)
    if isinstance(predictions, (list, tuple)):
        return np.asarray(predictions[0], dtype=np.float32)
    return np.asarray(predictions, dtype=np.float32)


def _build_multitask_targets(main_targets, aux_targets=None, main_horizon=None, required_aux_horizons=None):
    _ = aux_targets, main_horizon, required_aux_horizons
    return np.asarray(main_targets, dtype=int)


def _build_multitask_sample_weights(targets_dict):
    if isinstance(targets_dict, dict):
        if "main_output" in targets_dict:
            targets = targets_dict["main_output"]
        else:
            targets = next(iter(targets_dict.values()))
    else:
        targets = targets_dict
    sample_weight = compute_sample_weight(class_weight="balanced", y=np.asarray(targets, dtype=int))
    return np.asarray(sample_weight, dtype=np.float32)


def _tune_gate_thresholds(
    y_true,
    y_prob,
    evaluator,
    min_coverage=MIN_COVERAGE_RATIO,
    min_trade_count=None,
    conf_grid=None,
    edge_grid=None,
    evaluator_kwargs=None,
):
    if conf_grid is None:
        conf_grid = np.round(np.arange(0.15, 0.71, 0.02), 2)
    if edge_grid is None:
        edge_grid = np.round(np.arange(0.00, 0.11, 0.01), 2)
    if evaluator_kwargs is None:
        evaluator_kwargs = {}

    best = None
    fallback = None
    for conf in conf_grid:
        for edge in edge_grid:
            metrics = evaluator(
                y_true,
                y_prob,
                confidence_threshold=float(conf),
                min_confidence_edge=float(edge),
                **evaluator_kwargs,
            )
            score_tuple = _metrics_tuple(metrics)
            candidate = {
                "confidence_threshold": float(conf),
                "min_confidence_edge": float(edge),
                "metrics": metrics,
                "score_tuple": score_tuple,
            }
            if fallback is None or score_tuple > fallback["score_tuple"]:
                fallback = candidate
            if float(metrics.get("confidence_ge_threshold_ratio", 0.0)) < float(min_coverage):
                continue
            if min_trade_count is not None and int(metrics.get("trading_proxy", {}).get("trade_count", 0)) < int(min_trade_count):
                continue
            if best is None or score_tuple > best["score_tuple"]:
                best = candidate

    if best is not None:
        return best
    if min_trade_count is not None:
        return None
    return fallback


def build_model(input_shape, num_classes=NUM_CLASSES):
    inputs = Input(shape=input_shape, name="input_tensor")

    x = Conv1D(64, kernel_size=3, padding="same", activation="relu")(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.25)(x)

    x = LSTM(128, return_sequences=True)(x)
    x = Dropout(0.30)(x)

    attn = Attention()([x, x])
    x = Concatenate()([x, attn])
    x = GlobalAveragePooling1D()(x)

    x = Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = Dropout(0.30)(x)
    x = Dense(32, activation="relu")(x)

    main_output = Dense(num_classes, activation="softmax", name="main_output")(x)

    model = Model(inputs=inputs, outputs=main_output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _trade_metrics_from_predictions(
    y_true,
    y_pred,
    threshold=DEFAULT_CONFIDENCE_THRESHOLD,
    min_confidence_edge=DEFAULT_MIN_CONFIDENCE_EDGE,
    y_prob=None,
):
    trades = []
    for idx, pred_class in enumerate(y_pred):
        if y_prob is not None:
            prob = np.asarray(y_prob[idx], dtype=float)
            confidence = float(np.max(prob))
            sorted_prob = np.sort(prob)
            confidence_edge = float(sorted_prob[-1] - sorted_prob[-2]) if len(sorted_prob) >= 2 else confidence
            if confidence < threshold or confidence_edge < min_confidence_edge:
                continue

        actual = int(y_true[idx])
        if pred_class == 1:
            continue

        if pred_class == 2:
            pnl_r = (2.0 / 1.5) if actual == 2 else (-1.0 if actual == 0 else 0.0)
        else:
            pnl_r = (2.0 / 1.5) if actual == 0 else (-1.0 if actual == 2 else 0.0)

        trades.append(pnl_r)

    if not trades:
        return {
            "trade_count": 0,
            "win_rate": 0.0,
            "expectancy_r": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_r": 0.0,
            "equity_curve_tail_r": [],
        }

    trades = np.asarray(trades, dtype=float)
    equity_curve = trades.cumsum()
    peaks = np.maximum.accumulate(equity_curve)
    drawdowns = peaks - equity_curve
    gross_profit = trades[trades > 0].sum()
    gross_loss = -trades[trades < 0].sum()

    return {
        "trade_count": int(len(trades)),
        "win_rate": float((trades > 0).mean()),
        "expectancy_r": float(trades.mean()),
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else float("inf"),
        "max_drawdown_r": float(drawdowns.max()) if len(drawdowns) else 0.0,
        "equity_curve_tail_r": equity_curve[-100:].tolist(),
    }


def evaluate_predictions(
    y_true,
    y_prob,
    confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
    min_confidence_edge=DEFAULT_MIN_CONFIDENCE_EDGE,
):
    y_pred = np.argmax(y_prob, axis=1)
    y_conf = np.max(y_prob, axis=1)
    sorted_prob = np.sort(y_prob, axis=1)
    y_edge = sorted_prob[:, -1] - sorted_prob[:, -2]
    trade_mask = (y_conf >= confidence_threshold) & (y_edge >= min_confidence_edge)
    labels = [0, 1, 2]
    target_names = ["DOWN", "SIDEWAYS", "UP"]

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=target_names,
            zero_division=0,
            output_dict=True,
        ),
        "confidence_ge_threshold_ratio": float(trade_mask.mean()),
        "trading_proxy": _trade_metrics_from_predictions(
            y_true,
            y_pred,
            confidence_threshold,
            min_confidence_edge,
            y_prob,
        ),
    }
    return metrics


def _metrics_tuple(metrics: dict):
    trading = metrics["trading_proxy"]
    return (
        float(trading["expectancy_r"]),
        float(trading["profit_factor"]),
        -float(trading["max_drawdown_r"]),
        float(metrics["macro_f1"]),
        float(metrics["weighted_f1"]),
        float(metrics["confidence_ge_threshold_ratio"]),
    )


def _booster_feature_names(feature_columns):
    return (
        ["base_prob_down", "base_prob_sideways", "base_prob_up", "base_confidence", "base_confidence_gap"]
        + [f"last_{name}" for name in feature_columns]
    )


def _make_booster_features(last_step_features: np.ndarray, base_prob: np.ndarray) -> np.ndarray:
    base_prob = np.asarray(base_prob, dtype=float)
    if base_prob.shape[-1] != NUM_CLASSES:
        raise ValueError(f"Expected {NUM_CLASSES} base probabilities, got shape {base_prob.shape}.")
    confidence = float(np.max(base_prob))
    sorted_prob = np.sort(base_prob)
    confidence_gap = float(sorted_prob[-1] - sorted_prob[-2]) if len(sorted_prob) >= 2 else confidence
    last_step_features = np.asarray(last_step_features, dtype=float).reshape(-1)
    return np.concatenate([base_prob, np.array([confidence, confidence_gap], dtype=float), last_step_features], axis=0)


def _booster_gate_label(actual_class: int, base_prob: np.ndarray, threshold=DEFAULT_CONFIDENCE_THRESHOLD, min_edge=DEFAULT_MIN_CONFIDENCE_EDGE):
    base_prob = np.asarray(base_prob, dtype=float)
    base_pred = int(np.argmax(base_prob))
    confidence = float(np.max(base_prob))
    sorted_prob = np.sort(base_prob)
    confidence_gap = float(sorted_prob[-1] - sorted_prob[-2]) if len(sorted_prob) >= 2 else confidence
    if base_pred == int(actual_class) and base_pred != 1 and confidence >= threshold and confidence_gap >= min_edge:
        return BOOSTER_ENTER_CLASS
    return BOOSTER_SKIP_CLASS


def _trade_metrics_from_action_predictions(
    y_true,
    y_pred,
    base_pred=None,
    y_prob=None,
    confidence_threshold=None,
    min_confidence_edge=None,
):
    if confidence_threshold is None:
        confidence_threshold = 0.0
    if min_confidence_edge is None:
        min_confidence_edge = 0.0

    trades = []
    for idx, pred_class in enumerate(y_pred):
        if int(pred_class) != BOOSTER_ENTER_CLASS:
            continue
        if y_prob is not None:
            prob = np.asarray(y_prob[idx], dtype=float)
            confidence = float(np.max(prob))
            sorted_prob = np.sort(prob)
            confidence_edge = float(sorted_prob[-1] - sorted_prob[-2]) if len(sorted_prob) >= 2 else confidence
            if confidence < confidence_threshold or confidence_edge < min_confidence_edge:
                continue

        if base_pred is None:
            raise ValueError("base_pred is required for booster action trade metrics.")
        actual = int(y_true[idx])
        predicted_direction = int(base_pred[idx])
        if predicted_direction == 1:
            continue
        pnl_r = (2.0 / 1.5) if predicted_direction == actual else -1.0

        trades.append(pnl_r)

    if not trades:
        return {
            "trade_count": 0,
            "win_rate": 0.0,
            "expectancy_r": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_r": 0.0,
            "equity_curve_tail_r": [],
        }

    trades = np.asarray(trades, dtype=float)
    equity_curve = trades.cumsum()
    peaks = np.maximum.accumulate(equity_curve)
    drawdowns = peaks - equity_curve
    gross_profit = trades[trades > 0].sum()
    gross_loss = -trades[trades < 0].sum()

    return {
        "trade_count": int(len(trades)),
        "win_rate": float((trades > 0).mean()),
        "expectancy_r": float(trades.mean()),
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else float("inf"),
        "max_drawdown_r": float(drawdowns.max()) if len(drawdowns) else 0.0,
        "equity_curve_tail_r": equity_curve[-100:].tolist(),
    }


def evaluate_booster_predictions(
    y_true_base,
    y_true_booster,
    base_prob,
    y_prob,
    confidence_threshold=0.50,
    min_confidence_edge=0.04,
):
    y_pred = np.argmax(y_prob, axis=1)
    y_conf = np.max(y_prob, axis=1)
    sorted_prob = np.sort(y_prob, axis=1)
    y_edge = sorted_prob[:, -1] - sorted_prob[:, -2]
    trade_mask = (y_pred == BOOSTER_ENTER_CLASS) & (y_conf >= confidence_threshold) & (y_edge >= min_confidence_edge)
    labels = [0, 1]
    target_names = ["SKIP", "ENTER"]
    base_pred = np.argmax(base_prob, axis=1)

    metrics = {
        "accuracy": float(accuracy_score(y_true_booster, y_pred)),
        "macro_f1": float(f1_score(y_true_booster, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true_booster, y_pred, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true_booster, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true_booster, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true_booster, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(
            y_true_booster,
            y_pred,
            labels=labels,
            target_names=target_names,
            zero_division=0,
            output_dict=True,
        ),
        "confidence_ge_threshold_ratio": float(trade_mask.mean()),
        "trading_proxy": _trade_metrics_from_action_predictions(
            y_true_base,
            y_pred,
            base_pred=base_pred,
            y_prob=y_prob,
            confidence_threshold=confidence_threshold,
            min_confidence_edge=min_confidence_edge,
        ),
    }
    return metrics


def _build_booster_targets(y_true_base, base_prob, threshold=DEFAULT_CONFIDENCE_THRESHOLD, min_edge=DEFAULT_MIN_CONFIDENCE_EDGE):
    return np.asarray([
        _booster_gate_label(actual, prob, threshold=threshold, min_edge=min_edge)
        for actual, prob in zip(y_true_base, base_prob)
    ], dtype=int)


def _summarize_reports(reports):
    if not reports:
        return {}

    def avg(path):
        vals = []
        for r in reports:
            node = r
            for key in path:
                node = node[key]
            vals.append(float(node))
        return float(np.mean(vals))

    return {
        "mean_accuracy": avg(["accuracy"]),
        "mean_macro_f1": avg(["macro_f1"]),
        "mean_weighted_f1": avg(["weighted_f1"]),
        "mean_expectancy_r": avg(["trading_proxy", "expectancy_r"]),
        "mean_profit_factor": avg(["trading_proxy", "profit_factor"]),
        "mean_max_drawdown_r": avg(["trading_proxy", "max_drawdown_r"]),
        "mean_trade_count": avg(["trading_proxy", "trade_count"]),
        "mean_confidence_ge_threshold_ratio": avg(["confidence_ge_threshold_ratio"]),
    }


def _save_scaler_and_contract(paths: TrainPaths, feature_columns, scaler, label_params):
    scaler_include_name = _artifact_name("GeneratedScaler", paths.pair_slug, "mqh")
    scaler_include_path = os.path.join(paths.project_root, scaler_include_name)
    scaler_include_copy_path = os.path.join(paths.project_root, "MQL5", "Experts", "LSTM", scaler_include_name)
    dp.save_feature_contract(
        feature_columns,
        output_path=paths.feature_contract_path,
        file_copy_path=paths.feature_contract_file_path,
    )
    dp.save_label_config(
        label_params,
        output_path=paths.label_config_path,
        file_copy_path=paths.label_config_file_path,
    )
    dp.save_scaler_params(
        scaler,
        output_path=paths.scaler_path,
        file_copy_path=paths.scaler_file_path,
    )
    dp.save_scaler_include(
        scaler,
        output_path=scaler_include_path,
        file_copy_path=scaler_include_copy_path,
    )

    _save_json(
        paths.latest_pointer_path,
        {
            "run_id": paths.run_id,
            "run_dir": paths.run_dir,
            "report_path": paths.latest_report_path,
            "feature_contract_path": paths.feature_contract_path,
            "label_config_path": paths.label_config_path,
            "scaler_path": paths.scaler_path,
            "scaler_include_path": scaler_include_path,
        },
    )
    if _is_legacy_default_pair(paths.symbol, paths.timeframe):
        _save_json(
            os.path.join(paths.history_dir, LATEST_POINTER_NAME),
            {
                "run_id": paths.run_id,
                "run_dir": paths.run_dir,
                "report_path": paths.latest_report_path,
                "feature_contract_path": paths.feature_contract_path,
                "label_config_path": paths.label_config_path,
                "scaler_path": paths.scaler_path,
                "scaler_include_path": scaler_include_path,
            },
        )
        shutil.copy2(paths.feature_contract_file_path, os.path.join(paths.project_root, "MQL5", "Files", "feature_contract.json"))
        shutil.copy2(paths.label_config_file_path, os.path.join(paths.project_root, "MQL5", "Files", "label_config.json"))
        shutil.copy2(paths.scaler_file_path, os.path.join(paths.project_root, "MQL5", "Files", "scaler_params.csv"))
        shutil.copy2(scaler_include_path, os.path.join(paths.project_root, "GeneratedScaler.mqh"))
        shutil.copy2(scaler_include_copy_path, os.path.join(paths.project_root, "MQL5", "Experts", "LSTM", "GeneratedScaler.mqh"))


def _fit_and_eval_fold(features_df, targets, aux_targets_map, fold_spec, main_horizon, seq_length=SEQ_LENGTH, epochs=BASE_SWEEP_EPOCHS):
    total_rows = len(features_df)
    train_end = int(total_rows * fold_spec[0])
    val_end = int(total_rows * fold_spec[1])

    if val_end <= train_end + seq_length:
        return None

    train_features = features_df.iloc[:train_end]
    fold_features = features_df.iloc[:val_end]
    fold_targets = targets[:val_end]
    fold_aux_targets = {h: np.asarray(targets_arr[:val_end], dtype=int) for h, targets_arr in aux_targets_map.items()}

    scaler = dp.fit_scaler(train_features)
    scaled_fold = dp.transform_features(fold_features, scaler)
    sequence_payload = _build_multitask_sequence_payload(
        scaled_fold,
        fold_targets,
        fold_aux_targets,
        seq_length=seq_length,
    )
    X = sequence_payload["X"]
    y = sequence_payload["y_main"]
    end_indices = sequence_payload["end_indices"]
    train_mask, val_mask, _ = _sequence_masks(end_indices, train_end, val_end)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    aux_train_targets = {h: seq_targets[train_mask] for h, seq_targets in sequence_payload["aux_sequences"].items()}
    aux_val_targets = {h: seq_targets[val_mask] for h, seq_targets in sequence_payload["aux_sequences"].items()}

    if len(X_train) == 0 or len(X_val) == 0:
        return None

    model = build_model((X_train.shape[1], X_train.shape[2]))
    train_targets = y_train
    val_targets = y_val
    sample_weights = _build_multitask_sample_weights(train_targets)
    callbacks = [
        EarlyStopping(monitor="val_loss", mode="min", patience=12, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", mode="min", factor=0.5, patience=6, min_lr=1e-6, verbose=0),
    ]

    print(
        f"    Fitting fold train<{fold_spec[0]:.2f} val<{fold_spec[1]:.2f} "
        f"with {len(X_train)} train seq / {len(X_val)} val seq",
        flush=True,
    )
    history = model.fit(
        X_train,
        train_targets,
        validation_data=(X_val, val_targets),
        epochs=epochs,
        batch_size=64,
        callbacks=callbacks,
        sample_weight=sample_weights,
        verbose=1,
    )

    val_prob = _predict_main_prob(model, X_val)
    min_trade_count = max(MIN_TRADE_COUNT_FLOOR, int(len(y_val) * MIN_TRADE_COUNT_FLOOR_RATIO))
    tuned = _tune_gate_thresholds(
        y_val,
        val_prob,
        evaluate_predictions,
        min_coverage=BOOSTER_MIN_COVERAGE_RATIO,
        min_trade_count=min_trade_count,
    )
    if tuned is None:
        return None
    metrics = tuned["metrics"]
    metrics["tuned_confidence_threshold"] = tuned["confidence_threshold"]
    metrics["tuned_min_confidence_edge"] = tuned["min_confidence_edge"]
    metrics["train_end_row"] = train_end
    metrics["val_end_row"] = val_end
    metrics["epochs_ran"] = len(history.history.get("loss", []))
    metrics["class_histogram_train"] = _class_histogram(y_train)
    metrics["class_histogram_val"] = _class_histogram(y_val)
    metrics["aux_horizons"] = sorted(int(h) for h in aux_targets_map.keys())
    return metrics


def sweep_label_configs(csv_file, label_candidates):
    sweep_reports = []
    best_config = None
    best_metrics = None
    best_tuple = None

    fold_specs = [(0.50, 0.65), (0.65, 0.80)]

    for idx, label_params in enumerate(label_candidates, start=1):
        print(f"\n===== Label Sweep {idx}/{len(label_candidates)} =====", flush=True)
        print("Label params:", label_params, flush=True)

        features_df, targets, feature_columns, aux_targets = dp.preprocess_data(
            csv_file,
            label_params=label_params,
            aux_horizons=AUX_HORIZONS,
            return_aux_targets=True,
        )
        class_hist = _class_histogram(targets)
        min_class_support = max(1, int(len(targets) * MIN_CLASS_SUPPORT_RATIO))
        if any(int(v) < min_class_support for v in class_hist.values()):
            print(
                f"  Skipping label config because class support collapsed below {min_class_support} samples: {class_hist}",
                flush=True,
            )
            continue

        candidate_reports = []
        for fold_spec in fold_specs:
            print(f"  Training fold {fold_spec[0]:.2f}->{fold_spec[1]:.2f} ...", flush=True)
            fold_metrics = _fit_and_eval_fold(features_df, targets, aux_targets, fold_spec, int(label_params["future_horizon"]))
            if fold_metrics is not None:
                candidate_reports.append(fold_metrics)

        if not candidate_reports:
            print("No valid folds for this label config; skipping.", flush=True)
            continue

        summary = _summarize_reports(candidate_reports)
        summary_tuple = (
            float(summary["mean_expectancy_r"]),
            float(summary["mean_profit_factor"]),
            -float(summary["mean_max_drawdown_r"]),
            float(summary["mean_macro_f1"]),
            float(summary["mean_weighted_f1"]),
            float(summary["mean_confidence_ge_threshold_ratio"]),
            float(summary["mean_trade_count"]),
        )

        sweep_entry = {
            "label_params": label_params,
            "label_config_hash": dp.build_label_config_hash(label_params),
            "feature_contract_hash": _feature_contract_hash(feature_columns),
            "class_histogram": class_hist,
            "fold_reports": candidate_reports,
            "summary": summary,
            "summary_tuple": summary_tuple,
        }
        sweep_reports.append(sweep_entry)

        print("Sweep summary:", summary, flush=True)

        if best_tuple is None or summary_tuple > best_tuple:
            best_tuple = summary_tuple
            best_config = label_params
            best_metrics = sweep_entry

    return best_config, best_metrics, sweep_reports


def _load_checkpoint_if_possible(paths: TrainPaths, expected_state: dict):
    if not os.path.exists(paths.state_path):
        return None, 0, "cold_start", None, None

    saved_state = _load_json(paths.state_path)
    if not saved_state:
        return None, 0, "cold_start", None, None

    state_match = (
        saved_state.get("feature_contract_hash") == expected_state["feature_contract_hash"]
        and int(saved_state.get("sequence_length", -1)) == SEQ_LENGTH
        and int(saved_state.get("training_schema_version", -1)) == int(expected_state.get("training_schema_version", -1))
        and saved_state.get("model_family") == expected_state.get("model_family")
    )

    if not state_match:
        return None, 0, "cold_start", None, saved_state

    checkpoint_path = saved_state.get("checkpoint_path", paths.last_path)
    if not os.path.exists(checkpoint_path):
        checkpoint_path = paths.last_path if os.path.exists(paths.last_path) else paths.best_path

    if not os.path.exists(checkpoint_path):
        return None, 0, "cold_start", None, saved_state

    try:
        model = load_model(checkpoint_path)
        initial_epoch = int(saved_state.get("epoch_completed", 0))
        resume_source = "last_checkpoint" if os.path.basename(checkpoint_path) == os.path.basename(paths.last_path) else "best_checkpoint"
        return model, initial_epoch, resume_source, checkpoint_path, saved_state
    except Exception as exc:
        print(f"Could not resume from checkpoint {checkpoint_path}: {exc}")
        return None, 0, "cold_start", None, saved_state


def _final_label_report(features_df, targets, label_params):
    return {
        "label_params": label_params,
        "label_config_hash": dp.build_label_config_hash(label_params),
        "class_histogram": _class_histogram(targets),
        "feature_contract_hash": _feature_contract_hash(list(features_df.columns)),
    }


def _final_split(total_rows):
    train_end = int(total_rows * 0.75)
    val_end = int(total_rows * 0.90)
    return train_end, val_end


def _collect_booster_oof_data(features_df, targets, aux_targets_map, feature_columns, main_horizon, fold_specs=None, seq_length=SEQ_LENGTH):
    if fold_specs is None:
        fold_specs = [(0.50, 0.65), (0.65, 0.80)]

    records = []
    for fold_spec in fold_specs:
        train_end = int(len(features_df) * fold_spec[0])
        val_end = int(len(features_df) * fold_spec[1])
        if val_end <= train_end + seq_length:
            continue

        train_features = features_df.iloc[:train_end]
        fold_features = features_df.iloc[:val_end]
        fold_targets = targets[:val_end]
        fold_aux_targets = {h: np.asarray(targets_arr[:val_end], dtype=int) for h, targets_arr in aux_targets_map.items()}

        scaler = dp.fit_scaler(train_features)
        scaled_fold = dp.transform_features(fold_features, scaler)
        sequence_payload = _build_multitask_sequence_payload(
            scaled_fold,
            fold_targets,
            fold_aux_targets,
            seq_length=seq_length,
        )
        X = sequence_payload["X"]
        y = sequence_payload["y_main"]
        end_indices = sequence_payload["end_indices"]
        train_mask, val_mask, _ = _sequence_masks(end_indices, train_end, val_end)

        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        val_end_indices = end_indices[val_mask]
        aux_train_targets = {h: seq_targets[train_mask] for h, seq_targets in sequence_payload["aux_sequences"].items()}
        aux_val_targets = {h: seq_targets[val_mask] for h, seq_targets in sequence_payload["aux_sequences"].items()}

        if len(X_train) == 0 or len(X_val) == 0:
            continue

        model = build_model((X_train.shape[1], X_train.shape[2]))
        train_targets = y_train
        val_targets = y_val
        sample_weights = _build_multitask_sample_weights(train_targets)
        callbacks = [
            EarlyStopping(monitor="val_loss", mode="min", patience=10, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor="val_loss", mode="min", factor=0.5, patience=5, min_lr=1e-6, verbose=0),
        ]
        print(
            f"    Collecting booster OOF fold train<{fold_spec[0]:.2f} val<{fold_spec[1]:.2f} "
            f"with {len(X_train)} train seq / {len(X_val)} val seq",
            flush=True,
        )
        model.fit(
            X_train,
            train_targets,
            validation_data=(X_val, val_targets),
            epochs=BASE_SWEEP_EPOCHS,
            batch_size=64,
            callbacks=callbacks,
            sample_weight=sample_weights,
            verbose=1,
        )

        val_prob = _predict_main_prob(model, X_val)
        for idx in range(len(X_val)):
            base_prob = val_prob[idx]
            booster_features = _make_booster_features(X_val[idx, -1, :], base_prob)
            booster_target = _booster_gate_label(int(y_val[idx]), base_prob)
            records.append(
                {
                    "end_index": int(val_end_indices[idx]),
                    "booster_features": booster_features.tolist(),
                    "base_target": int(y_val[idx]),
                    "booster_target": int(booster_target),
                    "base_prob": base_prob.tolist(),
                    "confidence": float(np.max(base_prob)),
                    "confidence_gap": float(np.sort(base_prob)[-1] - np.sort(base_prob)[-2]) if len(base_prob) >= 2 else float(np.max(base_prob)),
                }
            )

    records.sort(key=lambda item: item["end_index"])
    return records


def _split_booster_records(records):
    if not records:
        return [], []
    split_index = max(1, int(len(records) * 0.70))
    split_index = min(split_index, len(records) - 1) if len(records) > 1 else 1
    return records[:split_index], records[split_index:]


def _fit_booster_candidate(X_train, y_train_booster, X_val, y_val_booster, y_val_base, candidate):
    booster = GradientBoostingClassifier(
        n_estimators=int(candidate["n_estimators"]),
        learning_rate=float(candidate["learning_rate"]),
        max_depth=int(candidate["max_depth"]),
        min_samples_leaf=int(candidate["min_samples_leaf"]),
        subsample=float(candidate["subsample"]),
        random_state=42,
    )
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train_booster)
    booster.fit(X_train, y_train_booster, sample_weight=sample_weight)
    val_prob = booster.predict_proba(X_val)
    min_trade_count = max(MIN_TRADE_COUNT_FLOOR, int(len(y_val_booster) * MIN_TRADE_COUNT_FLOOR_RATIO))
    tuned = _tune_gate_thresholds(
        y_val_base,
        val_prob,
        lambda _y_true, y_prob, confidence_threshold, min_confidence_edge: evaluate_booster_predictions(
            y_val_base,
            y_val_booster,
            X_val[:, :NUM_CLASSES],
            y_prob,
            confidence_threshold=confidence_threshold,
            min_confidence_edge=min_confidence_edge,
        ),
        min_coverage=BOOSTER_MIN_COVERAGE_RATIO,
        min_trade_count=min_trade_count,
    )
    if tuned is None:
        return None, None
    metrics = tuned["metrics"]
    metrics["tuned_confidence_threshold"] = tuned["confidence_threshold"]
    metrics["tuned_min_confidence_edge"] = tuned["min_confidence_edge"]
    return booster, metrics


def _prepare_final_sequences(features_df, targets, aux_targets_map, feature_columns):
    train_end, val_end = _final_split(len(features_df))
    train_features = features_df.iloc[:train_end]
    scaler = dp.fit_scaler(train_features)
    scaled_all = dp.transform_features(features_df, scaler)
    sequence_payload = _build_multitask_sequence_payload(
        scaled_all,
        targets,
        aux_targets_map,
        seq_length=SEQ_LENGTH,
    )
    X_all = sequence_payload["X"]
    y_all = sequence_payload["y_main"]
    end_indices = sequence_payload["end_indices"]
    train_mask, val_mask, test_mask = _sequence_masks(end_indices, train_end, val_end)
    aux_sequences = sequence_payload["aux_sequences"]

    payload = {
        "scaler": scaler,
        "train_end": int(train_end),
        "val_end": int(val_end),
        "X_train": X_all[train_mask],
        "y_train": y_all[train_mask],
        "X_val": X_all[val_mask],
        "y_val": y_all[val_mask],
        "X_test": X_all[test_mask],
        "y_test": y_all[test_mask],
        "end_indices": end_indices,
        "aux_train": {h: targets_arr[train_mask] for h, targets_arr in aux_sequences.items()},
        "aux_val": {h: targets_arr[val_mask] for h, targets_arr in aux_sequences.items()},
        "aux_test": {h: targets_arr[test_mask] for h, targets_arr in aux_sequences.items()},
    }
    if len(payload["X_train"]) == 0 or len(payload["X_val"]) == 0 or len(payload["X_test"]) == 0:
        raise ValueError("One of the train/val/test splits is empty. Increase history size or adjust split ratios.")
    return payload


def _evaluate_combined_stack(
    base_model,
    booster_model,
    stack_payload,
    base_thresholds=None,
    booster_thresholds=None,
):
    if base_thresholds is None:
        base_thresholds = {"confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD, "min_confidence_edge": DEFAULT_MIN_CONFIDENCE_EDGE}
    if booster_thresholds is None:
        booster_thresholds = {"confidence_threshold": 0.50, "min_confidence_edge": 0.04}

    base_val_prob = _predict_main_prob(base_model, stack_payload["X_val"])
    base_test_prob = _predict_main_prob(base_model, stack_payload["X_test"])

    base_validation_metrics = evaluate_predictions(
        stack_payload["y_val"],
        base_val_prob,
        confidence_threshold=base_thresholds["confidence_threshold"],
        min_confidence_edge=base_thresholds["min_confidence_edge"],
    )
    base_test_metrics = evaluate_predictions(
        stack_payload["y_test"],
        base_test_prob,
        confidence_threshold=base_thresholds["confidence_threshold"],
        min_confidence_edge=base_thresholds["min_confidence_edge"],
    )

    if booster_model is None:
        return {
            "base_validation_metrics": base_validation_metrics,
            "base_test_metrics": base_test_metrics,
            "combined_validation_metrics": base_validation_metrics,
            "combined_test_metrics": base_test_metrics,
            "base_val_prob": base_val_prob,
            "base_test_prob": base_test_prob,
            "booster_val_prob": None,
            "booster_test_prob": None,
            "booster_val_features": None,
            "booster_test_features": None,
            "booster_val_target": None,
            "booster_test_target": None,
            "base_thresholds": base_thresholds,
            "booster_thresholds": booster_thresholds,
        }

    booster_val_features = np.array([
        _make_booster_features(stack_payload["X_val"][i, -1, :], base_val_prob[i])
        for i in range(len(stack_payload["X_val"]))
    ], dtype=np.float32)
    booster_test_features = np.array([
        _make_booster_features(stack_payload["X_test"][i, -1, :], base_test_prob[i])
        for i in range(len(stack_payload["X_test"]))
    ], dtype=np.float32)

    booster_val_target = _build_booster_targets(stack_payload["y_val"], base_val_prob)
    booster_test_target = _build_booster_targets(stack_payload["y_test"], base_test_prob)

    booster_val_prob = booster_model.predict_proba(booster_val_features)
    booster_test_prob = booster_model.predict_proba(booster_test_features)

    combined_validation_metrics = evaluate_booster_predictions(
        stack_payload["y_val"],
        booster_val_target,
        base_val_prob,
        booster_val_prob,
        confidence_threshold=booster_thresholds["confidence_threshold"],
        min_confidence_edge=booster_thresholds["min_confidence_edge"],
    )
    combined_test_metrics = evaluate_booster_predictions(
        stack_payload["y_test"],
        booster_test_target,
        base_test_prob,
        booster_test_prob,
        confidence_threshold=booster_thresholds["confidence_threshold"],
        min_confidence_edge=booster_thresholds["min_confidence_edge"],
    )

    return {
        "base_validation_metrics": base_validation_metrics,
        "base_test_metrics": base_test_metrics,
        "combined_validation_metrics": combined_validation_metrics,
        "combined_test_metrics": combined_test_metrics,
        "base_val_prob": base_val_prob,
        "base_test_prob": base_test_prob,
        "booster_val_prob": booster_val_prob,
        "booster_test_prob": booster_test_prob,
        "booster_val_features": booster_val_features,
        "booster_test_features": booster_test_features,
        "booster_val_target": booster_val_target,
        "booster_test_target": booster_test_target,
        "base_thresholds": base_thresholds,
        "booster_thresholds": booster_thresholds,
    }


def _train_booster_from_oof(records, feature_columns):
    if not records:
        raise ValueError("No OOF records were collected for booster training.")

    train_records, val_records = _split_booster_records(records)
    if not train_records or not val_records:
        raise ValueError("Not enough OOF records to train and validate the booster.")

    X_train = np.asarray([row["booster_features"] for row in train_records], dtype=np.float32)
    y_train = np.asarray([row["booster_target"] for row in train_records], dtype=int)
    X_val = np.asarray([row["booster_features"] for row in val_records], dtype=np.float32)
    y_val_booster = np.asarray([row["booster_target"] for row in val_records], dtype=int)
    y_val_base = np.asarray([row["base_target"] for row in val_records], dtype=int)

    best_booster = None
    best_candidate = None
    best_metrics = None
    best_tuple = None

    for candidate in BOOSTER_CANDIDATES:
        booster, metrics = _fit_booster_candidate(X_train, y_train, X_val, y_val_booster, y_val_base, candidate)
        if booster is None or metrics is None:
            print(f"  Booster candidate {candidate} rejected by coverage floor.", flush=True)
            continue
        metric_tuple = _metrics_tuple(metrics)
        print(f"  Booster candidate {candidate} -> {metric_tuple}", flush=True)
        if best_tuple is None or metric_tuple > best_tuple:
            best_tuple = metric_tuple
            best_booster = booster
            best_candidate = candidate
            best_metrics = metrics

    if best_booster is None:
        raise ValueError("No booster candidate satisfied the minimum trade coverage constraint.")

    return {
        "model": best_booster,
        "params": best_candidate,
        "validation_metrics": best_metrics,
        "validation_tuple": best_tuple,
        "train_rows": int(len(train_records)),
        "val_rows": int(len(val_records)),
        "feature_names": _booster_feature_names(feature_columns),
    }


def _export_booster_onnx(model, booster_feature_names, paths: TrainPaths):
    if convert_sklearn is None or FloatTensorType is None or onnx is None:
        raise RuntimeError(
            "skl2onnx/onnx are required to export the booster model. Re-run Run_Training.bat with the dependency installed."
        )

    initial_types = [("input_tensor", FloatTensorType([None, len(booster_feature_names)]))]
    onnx_model = convert_sklearn(
        model,
        initial_types=initial_types,
        target_opset=13,
        options={id(model): {"zipmap": False}},
    )

    probability_output_name = None
    for output in onnx_model.graph.output:
        if "prob" in output.name.lower():
            probability_output_name = output.name
            break
    if probability_output_name is None and len(onnx_model.graph.output) > 1:
        probability_output_name = onnx_model.graph.output[-1].name

    if probability_output_name is not None and len(onnx_model.graph.output) > 1:
        keep_output = None
        for output in onnx_model.graph.output:
            if output.name == probability_output_name:
                keep_output = output
                break
        if keep_output is not None:
            del onnx_model.graph.output[:]
            onnx_model.graph.output.extend([keep_output])

    onnx.checker.check_model(onnx_model)
    _ensure_parent(paths.booster_onnx_path)
    onnx.save_model(onnx_model, paths.booster_onnx_path)
    shutil.copy2(paths.booster_onnx_path, os.path.join(paths.project_root, "MQL5_EA", os.path.basename(paths.booster_onnx_path)))


def _train_final_model(features_df, targets, aux_targets_map, feature_columns, label_params, paths: TrainPaths, resume_enabled=True):
    train_end, val_end = _final_split(len(features_df))
    if val_end <= train_end + SEQ_LENGTH:
        raise ValueError("Not enough data for final training split.")

    train_features = features_df.iloc[:train_end]
    scaler = dp.fit_scaler(train_features)
    scaled_all = dp.transform_features(features_df, scaler)
    sequence_payload = _build_multitask_sequence_payload(
        scaled_all,
        targets,
        aux_targets_map,
        seq_length=SEQ_LENGTH,
    )
    X_all = sequence_payload["X"]
    y_all = sequence_payload["y_main"]
    end_indices = sequence_payload["end_indices"]
    train_mask, val_mask, test_mask = _sequence_masks(end_indices, train_end, val_end)
    aux_sequences = sequence_payload["aux_sequences"]

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_val, y_val = X_all[val_mask], y_all[val_mask]
    X_test, y_test = X_all[test_mask], y_all[test_mask]

    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        raise ValueError("One of the train/val/test splits is empty. Increase history size or adjust split ratios.")

    _ensure_parent(paths.best_path)
    expected_state = {
        "feature_contract_hash": _feature_contract_hash(feature_columns),
        "sequence_length": SEQ_LENGTH,
        "training_schema_version": TRAINING_SCHEMA_VERSION,
        "model_family": MODEL_FAMILY,
    }
    history_records = _load_jsonl(paths.index_path)
    previous_record = history_records[-1] if history_records else None

    resumed_model = None
    initial_epoch = 0
    resume_source = "cold_start"
    resume_checkpoint_path = None
    saved_state = None

    if resume_enabled:
        resumed_model, initial_epoch, resume_source, resume_checkpoint_path, saved_state = _load_checkpoint_if_possible(paths, expected_state)
        if resumed_model is None:
            resumed_model, initial_epoch, resume_source, resume_checkpoint_path, saved_state = _load_resume_checkpoint_from_history(
                history_records,
                expected_state,
            )

    main_horizon = int(label_params.get("future_horizon", 24))
    model = resumed_model if resumed_model is not None else build_model((X_train.shape[1], X_train.shape[2]))

    target_epochs = max(BASE_FINAL_EPOCHS, initial_epoch + 120)
    class_histogram = _class_histogram(targets)
    train_targets = y_train
    val_targets = y_val
    sample_weights = _build_multitask_sample_weights(train_targets)
    best_validation_metrics = None
    best_validation_score_tuple = None
    best_validation_loss = None
    best_epoch = initial_epoch
    wait_epochs = 0
    lr_wait_epochs = 0
    current_lr = None

    def state_payload_factory(epoch_completed, logs):
        return {
            "run_id": paths.run_id,
            "run_dir": paths.run_dir,
            "resume_source": resume_source,
            "resume_enabled": bool(resume_enabled),
            "checkpoint_path": paths.last_path,
            "resume_checkpoint_path": resume_checkpoint_path,
            "best_checkpoint_path": paths.best_path,
            "feature_contract_hash": expected_state["feature_contract_hash"],
            "training_schema_version": TRAINING_SCHEMA_VERSION,
            "model_family": MODEL_FAMILY,
            "label_config_hash": dp.build_label_config_hash(label_params),
            "sequence_length": SEQ_LENGTH,
            "feature_count": len(feature_columns),
            "label_params": label_params,
            "aux_horizons": sorted(int(h) for h in aux_targets_map.keys()),
            "class_histogram": class_histogram,
            "epoch_completed": int(epoch_completed),
            "best_epoch": int(best_epoch),
            "best_validation_score_tuple": list(best_validation_score_tuple) if best_validation_score_tuple is not None else None,
            "best_validation_loss": float(best_validation_loss) if best_validation_loss is not None else None,
            "best_validation_metrics": best_validation_metrics,
            "last_logs": {k: float(v) for k, v in logs.items() if isinstance(v, (int, float, np.floating))},
        }

    class TrainingStateCallback(tf.keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            nonlocal current_lr
            try:
                current_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
            except Exception:
                current_lr = None

        def on_epoch_end(self, epoch, logs=None):
            nonlocal best_validation_metrics, best_validation_score_tuple, best_validation_loss, best_epoch, wait_epochs, lr_wait_epochs, current_lr
            logs = logs or {}
            val_prob = _predict_main_prob(self.model, X_val)
            tuned = _tune_gate_thresholds(
                y_val,
                val_prob,
                evaluate_predictions,
                min_coverage=BOOSTER_MIN_COVERAGE_RATIO,
                min_trade_count=max(MIN_TRADE_COUNT_FLOOR, int(len(y_val) * MIN_TRADE_COUNT_FLOOR_RATIO)),
            )
            if tuned is None:
                metrics = evaluate_predictions(
                    y_val,
                    val_prob,
                    confidence_threshold=1.0,
                    min_confidence_edge=1.0,
                )
                metrics["tuned_confidence_threshold"] = None
                metrics["tuned_min_confidence_edge"] = None
                score_tuple = (-1e9, -1e9, -1e9, -1e9, -1e9, -1e9)
                self.model.save(paths.last_path)
                _save_json(paths.state_path, state_payload_factory(initial_epoch + epoch + 1, logs))
                wait_epochs += 1
                lr_wait_epochs += 1
                if wait_epochs >= ADAPTIVE_PATIENCE and (initial_epoch + epoch + 1) >= ADAPTIVE_MIN_EPOCHS:
                    print(f"  Adaptive stop: no score improvement for {wait_epochs} epoch(s).", flush=True)
                    self.model.stop_training = True
                return
            metrics = tuned["metrics"]
            metrics["tuned_confidence_threshold"] = tuned["confidence_threshold"]
            metrics["tuned_min_confidence_edge"] = tuned["min_confidence_edge"]
            score_tuple = _metrics_tuple(metrics)
            val_loss = float(logs.get("val_loss", 0.0))

            self.model.save(paths.last_path)

            if best_validation_score_tuple is None or tuple(score_tuple) > tuple(best_validation_score_tuple):
                best_validation_metrics = metrics
                best_validation_score_tuple = score_tuple
                best_validation_loss = val_loss
                best_epoch = initial_epoch + epoch + 1
                wait_epochs = 0
                lr_wait_epochs = 0
                self.model.save(paths.best_path)
                print(f"  New best at epoch {best_epoch}: score={score_tuple}", flush=True)
            else:
                wait_epochs += 1
                lr_wait_epochs += 1

            if current_lr is not None and lr_wait_epochs >= ADAPTIVE_LR_PATIENCE:
                try:
                    current_lr = max(current_lr * 0.5, 1e-6)
                    tf.keras.backend.set_value(self.model.optimizer.learning_rate, current_lr)
                    lr_wait_epochs = 0
                    print(f"  Reducing learning rate to {current_lr:.8f}", flush=True)
                except Exception as exc:
                    print(f"  Could not adjust learning rate: {exc}", flush=True)

            _save_json(paths.state_path, state_payload_factory(initial_epoch + epoch + 1, logs))

            if wait_epochs >= ADAPTIVE_PATIENCE and (initial_epoch + epoch + 1) >= ADAPTIVE_MIN_EPOCHS:
                print(f"  Adaptive stop: no score improvement for {wait_epochs} epoch(s).", flush=True)
                self.model.stop_training = True

    print(f"Training final model. Resume source: {resume_source}", flush=True)
    history = model.fit(
        X_train,
        train_targets,
        validation_data=(X_val, val_targets),
        epochs=target_epochs,
        initial_epoch=initial_epoch,
        batch_size=64,
        callbacks=[TrainingStateCallback()],
        sample_weight=sample_weights,
        verbose=1,
    )

    best_model = load_model(paths.best_path) if os.path.exists(paths.best_path) else model
    main_best_model = Model(inputs=best_model.input, outputs=best_model.get_layer("main_output").output)
    val_prob = _predict_main_prob(main_best_model, X_val)
    test_prob = _predict_main_prob(main_best_model, X_test)

    if best_validation_metrics is None:
        tuned = _tune_gate_thresholds(
            y_val,
            val_prob,
            evaluate_predictions,
            min_coverage=BOOSTER_MIN_COVERAGE_RATIO,
            min_trade_count=max(MIN_TRADE_COUNT_FLOOR, int(len(y_val) * MIN_TRADE_COUNT_FLOOR_RATIO)),
        )
        if tuned is None:
            raise ValueError("Final base model could not satisfy the minimum trade coverage constraint.")
        best_validation_metrics = tuned["metrics"]
        best_validation_metrics["tuned_confidence_threshold"] = tuned["confidence_threshold"]
        best_validation_metrics["tuned_min_confidence_edge"] = tuned["min_confidence_edge"]
        best_validation_score_tuple = _metrics_tuple(best_validation_metrics)
        best_validation_loss = float(history.history["val_loss"][-1]) if history.history.get("val_loss") else None
        best_epoch = initial_epoch + len(history.history.get("loss", []))

    comparison_block = _comparison_block(
        {
            "best_validation_score_tuple": list(best_validation_score_tuple) if best_validation_score_tuple is not None else [],
            "best_validation_metrics": best_validation_metrics,
            "test_metrics": evaluate_predictions(
                y_test,
                test_prob,
                confidence_threshold=best_validation_metrics.get("tuned_confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD),
                min_confidence_edge=best_validation_metrics.get("tuned_min_confidence_edge", DEFAULT_MIN_CONFIDENCE_EDGE),
            ),
        },
        previous_record,
    )

    train_report = {
        "run_id": paths.run_id,
        "run_dir": paths.run_dir,
        "history_dir": paths.history_dir,
        "index_path": paths.index_path,
        "latest_pointer_path": paths.latest_pointer_path,
        "best_registry_path": paths.best_registry_path,
        "resume_source": resume_source,
        "resumed_from_checkpoint": resume_source != "cold_start",
        "checkpoint_dir": paths.checkpoint_dir,
        "checkpoint_path": paths.last_path,
        "resume_checkpoint_path": resume_checkpoint_path,
        "best_checkpoint_path": paths.best_path,
        "state_path": paths.state_path,
        "feature_contract": _feature_contract(feature_columns),
        "feature_contract_hash": expected_state["feature_contract_hash"],
        "training_schema_version": TRAINING_SCHEMA_VERSION,
        "model_family": MODEL_FAMILY,
        "label_config": label_params,
        "label_config_hash": dp.build_label_config_hash(label_params),
        "class_histogram": class_histogram,
        "best_validation_epoch": int(best_epoch),
        "best_validation_score_tuple": list(best_validation_score_tuple) if best_validation_score_tuple is not None else None,
        "best_validation_loss": float(best_validation_loss) if best_validation_loss is not None else None,
        "best_validation_metrics": best_validation_metrics,
        "training_history": {
            "epochs_ran": len(history.history.get("loss", [])),
            "final_loss": float(history.history["loss"][-1]),
            "final_val_loss": float(history.history["val_loss"][-1]),
            "final_accuracy": float(history.history["accuracy"][-1]) if "accuracy" in history.history else None,
            "final_val_accuracy": float(history.history["val_accuracy"][-1]) if "val_accuracy" in history.history else None,
            "start_epoch": int(initial_epoch),
            "target_epochs": int(target_epochs),
        },
        "validation_metrics": best_validation_metrics,
        "test_metrics": evaluate_predictions(
            y_test,
            test_prob,
            confidence_threshold=best_validation_metrics.get("tuned_confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD),
            min_confidence_edge=best_validation_metrics.get("tuned_min_confidence_edge", DEFAULT_MIN_CONFIDENCE_EDGE),
        ),
        "final_sample_weights": float(sample_weights.mean()) if hasattr(sample_weights, "mean") else None,
        "comparison_to_previous": comparison_block,
        "aux_horizons": sorted(int(h) for h in aux_targets_map.keys()),
    }

    train_report["selection_reason"] = (
        "selected by best walk-forward expectancy/profit factor with macro F1 tie-breaker, subject to coverage floor; training resumes from best compatible history when available"
    )

    _save_json(paths.report_path, train_report)
    _save_json(paths.latest_report_path, train_report)
    _save_json(paths.latest_file_report_path, train_report)
    _save_json(paths.state_path, {
        "run_id": paths.run_id,
        "run_dir": paths.run_dir,
        "resume_source": resume_source,
        "resume_enabled": bool(resume_enabled),
        "checkpoint_path": paths.last_path,
        "resume_checkpoint_path": resume_checkpoint_path,
        "best_checkpoint_path": paths.best_path,
        "feature_contract_hash": expected_state["feature_contract_hash"],
        "label_config_hash": dp.build_label_config_hash(label_params),
        "sequence_length": SEQ_LENGTH,
        "training_schema_version": TRAINING_SCHEMA_VERSION,
        "model_family": MODEL_FAMILY,
        "feature_count": len(feature_columns),
        "label_params": label_params,
        "aux_horizons": sorted(int(h) for h in aux_targets_map.keys()),
        "class_histogram": class_histogram,
        "epoch_completed": int(initial_epoch + len(history.history.get("loss", []))),
        "best_epoch": int(best_epoch),
        "best_validation_score_tuple": list(best_validation_score_tuple) if best_validation_score_tuple is not None else None,
        "best_validation_loss": float(best_validation_loss) if best_validation_loss is not None else None,
        "best_validation_metrics": best_validation_metrics,
        "final_metrics": {
            "validation": train_report["validation_metrics"]["trading_proxy"],
            "test": train_report["test_metrics"]["trading_proxy"],
        },
    })

    run_record = {
        "run_id": paths.run_id,
        "run_dir": paths.run_dir,
        "timestamp_utc": paths.run_id.split("_")[0],
        "feature_contract_hash": expected_state["feature_contract_hash"],
        "label_config_hash": dp.build_label_config_hash(label_params),
        "sequence_length": SEQ_LENGTH,
        "training_schema_version": TRAINING_SCHEMA_VERSION,
        "model_family": MODEL_FAMILY,
        "label_params": label_params,
        "aux_horizons": sorted(int(h) for h in AUX_HORIZONS),
        "best_checkpoint_path": paths.best_path,
        "last_checkpoint_path": paths.last_path,
        "state_path": paths.state_path,
        "report_path": paths.report_path,
        "latest_report_path": paths.latest_report_path,
        "best_validation_score_tuple": list(best_validation_score_tuple) if best_validation_score_tuple is not None else None,
        "best_validation_metrics": best_validation_metrics,
        "test_metrics": train_report["test_metrics"],
        "validation_metrics": train_report["validation_metrics"],
        "training_history": train_report["training_history"],
        "resume_source": resume_source,
        "resume_checkpoint_path": resume_checkpoint_path,
        "comparison_to_previous": comparison_block,
        "selection_reason": train_report["selection_reason"],
        "class_histogram": class_histogram,
    }
    _append_jsonl(paths.index_path, run_record)
    _save_json(paths.latest_pointer_path, {
        "run_id": paths.run_id,
        "run_dir": paths.run_dir,
        "report_path": paths.latest_report_path,
        "best_checkpoint_path": paths.best_path,
        "state_path": paths.state_path,
    })
    _update_best_registry(paths.best_registry_path, run_record)

    return model, history, train_report, scaler


def _export_onnx(model, feature_columns, paths: TrainPaths):
    input_signature = [tf.TensorSpec([None, SEQ_LENGTH, len(feature_columns)], tf.float32, name="input_tensor")]
    tf2onnx.convert.from_keras(
        model,
        input_signature=input_signature,
        opset=13,
        output_path=paths.onnx_path,
    )
    shutil.copy2(paths.onnx_path, os.path.join(paths.project_root, "MQL5_EA", os.path.basename(paths.onnx_path)))


if __name__ == "__main__":
    args = _parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    symbol = _normalize_symbol(args.symbol)
    timeframe = _normalize_timeframe(args.timeframe)
    csv_file = os.path.abspath(args.csv) if args.csv else os.path.join(project_root, _default_csv_name(symbol, timeframe))

    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"{csv_file} not found.")

    strict_eurusd_baseline = bool(args.strict_eurusd_baseline and symbol != DEFAULT_SYMBOL)
    if strict_eurusd_baseline:
        sweep_best_label_params = _load_strict_eurusd_label_params(project_root)
        sweep_best_metrics = None
        sweep_reports = []
        sweep_best_summary = None
        selection_reason = "strict EURUSD baseline mode: reused final EURUSD label configuration without pair-specific label sweep"
        print("Selected label params (strict EURUSD baseline):", sweep_best_label_params, flush=True)
    else:
        sweep_best_label_params, sweep_best_metrics, sweep_reports = sweep_label_configs(csv_file, LABEL_CANDIDATES)
        if sweep_best_label_params is None:
            raise RuntimeError("No valid label configuration produced usable folds.")

        sweep_best_summary = sweep_best_metrics["summary"]
        selection_reason = (
            "best walk-forward tuple (expectancy_r, profit_factor, -max_drawdown_r, macro_f1, weighted_f1, coverage_ratio) with coverage floor"
        )
        print("Selected label params:", sweep_best_label_params, flush=True)
        print("Selected walk-forward summary:", sweep_best_summary, flush=True)

    features_df, targets, feature_columns, aux_targets = dp.preprocess_data(
        csv_file,
        label_params=sweep_best_label_params,
        aux_horizons=AUX_HORIZONS,
        return_aux_targets=True,
    )

    feature_contract_hash = _feature_contract_hash(feature_columns)
    label_config_hash = dp.build_label_config_hash(sweep_best_label_params)
    run_id = _now_run_id(feature_contract_hash, label_config_hash, symbol, timeframe)
    paths = _paths_for(project_root, run_id, feature_contract_hash, label_config_hash, symbol, timeframe)
    label_params_with_aux = dict(sweep_best_label_params)
    label_params_with_aux["aux_horizons"] = list(AUX_HORIZONS)

    _save_scaler_and_contract(
        paths,
        feature_columns,
        dp.fit_scaler(features_df.iloc[:_final_split(len(features_df))[0]]),
        label_params_with_aux,
    )

    model, history, train_report, scaler = _train_final_model(
        features_df,
        targets,
        aux_targets,
        feature_columns,
        sweep_best_label_params,
        paths,
        resume_enabled=True,
    )

    base_best_model = load_model(paths.best_path) if os.path.exists(paths.best_path) else model
    base_inference_model = Model(inputs=base_best_model.input, outputs=base_best_model.get_layer("main_output").output)
    base_threshold_config = {
        "confidence_threshold": float(train_report["best_validation_metrics"].get("tuned_confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD)),
        "min_confidence_edge": float(train_report["best_validation_metrics"].get("tuned_min_confidence_edge", DEFAULT_MIN_CONFIDENCE_EDGE)),
        "source": "base_validation_metrics",
    }
    booster_records = _collect_booster_oof_data(features_df, targets, aux_targets, feature_columns, int(sweep_best_label_params["future_horizon"]))
    booster_stack = None
    booster_error = None
    booster_feature_names = _booster_feature_names(feature_columns)
    booster_train_records, booster_val_records = _split_booster_records(booster_records) if booster_records else ([], [])
    try:
        booster_stack = _train_booster_from_oof(booster_records, feature_columns)
    except ValueError as exc:
        booster_error = str(exc)
        print(f"  Booster stage skipped: {booster_error}", flush=True)

    stack_payload = _prepare_final_sequences(features_df, targets, aux_targets, feature_columns)
    combined_stack = _evaluate_combined_stack(
        base_inference_model,
        booster_stack["model"] if booster_stack is not None else None,
        stack_payload,
        base_thresholds=base_threshold_config,
        booster_thresholds=(
            {
                "confidence_threshold": booster_stack["validation_metrics"].get("tuned_confidence_threshold", 0.50),
                "min_confidence_edge": booster_stack["validation_metrics"].get("tuned_min_confidence_edge", 0.04),
            }
            if booster_stack is not None else None
        ),
    )

    base_validation_metrics = combined_stack["base_validation_metrics"]
    base_test_metrics = combined_stack["base_test_metrics"]
    combined_validation_metrics = combined_stack["combined_validation_metrics"]
    combined_test_metrics = combined_stack["combined_test_metrics"]
    min_final_stack_trade_count = max(
        MIN_TRADE_COUNT_FLOOR,
        int(len(stack_payload["y_val"]) * MIN_TRADE_COUNT_FLOOR_RATIO),
    )

    if booster_stack is not None:
        combined_validation_trade_count = int(combined_validation_metrics.get("trading_proxy", {}).get("trade_count", 0))
        if combined_validation_trade_count < min_final_stack_trade_count:
            booster_error = (
                f"Combined booster stack produced only {combined_validation_trade_count} validation trades, "
                f"below coverage floor {min_final_stack_trade_count}; using base LSTM only."
            )
            print(f"  Booster stage skipped: {booster_error}", flush=True)
            booster_stack = None
            combined_validation_metrics = base_validation_metrics
            combined_test_metrics = base_test_metrics

    _export_onnx(base_inference_model, feature_columns, paths)
    if booster_stack is not None:
        _export_booster_onnx(booster_stack["model"], booster_stack["feature_names"], paths)

    train_report["base_validation_metrics"] = base_validation_metrics
    train_report["base_test_metrics"] = base_test_metrics
    if booster_stack is not None:
        train_report["booster"] = {
            "enabled": True,
            "status": "trained",
            "feature_count": BOOSTER_INPUT_FEATURE_COUNT,
            "feature_columns": booster_stack["feature_names"],
            "params": booster_stack["params"],
            "validation_metrics": booster_stack["validation_metrics"],
            "validation_tuple": list(booster_stack["validation_tuple"]) if booster_stack["validation_tuple"] is not None else None,
            "train_rows": booster_stack["train_rows"],
            "val_rows": booster_stack["val_rows"],
            "onnx_path": paths.booster_onnx_path,
        }
    else:
        train_report["booster"] = {
            "enabled": False,
            "status": "skipped_degenerate_final_gate" if booster_error and "Combined booster stack produced only" in booster_error else "skipped_no_candidate",
            "reason": booster_error,
            "feature_count": BOOSTER_INPUT_FEATURE_COUNT,
            "feature_columns": booster_feature_names,
            "params": None,
            "validation_metrics": None,
            "validation_tuple": None,
            "train_rows": int(len(booster_train_records)),
            "val_rows": int(len(booster_val_records)),
            "onnx_path": None,
        }
    train_report["threshold_config"] = {
        "base": dict(base_threshold_config),
        "booster": (
            {
                "confidence_threshold": float(booster_stack["validation_metrics"].get("tuned_confidence_threshold", 0.50)),
                "min_confidence_edge": float(booster_stack["validation_metrics"].get("tuned_min_confidence_edge", 0.04)),
                "source": "booster_validation_metrics",
            }
            if booster_stack is not None else None
        ),
        "recommended_signal_mode": "with_booster" if booster_stack is not None else "base_only",
    }
    train_report["validation_metrics"] = combined_validation_metrics
    train_report["test_metrics"] = combined_test_metrics
    train_report["best_validation_metrics"] = combined_validation_metrics
    train_report["best_validation_score_tuple"] = list(_metrics_tuple(combined_validation_metrics))
    train_report["final_stack_metrics"] = {
        "base_validation_metrics": base_validation_metrics,
        "base_test_metrics": base_test_metrics,
        "combined_validation_metrics": combined_validation_metrics,
        "combined_test_metrics": combined_test_metrics,
    }
    if booster_stack is not None:
        train_report["stack_selection_reason"] = (
            "multi-horizon base LSTM predicts sequences; booster ONNX filters actions using OOF meta-features, tuned thresholds, and final stack metrics"
        )
    else:
        train_report["stack_selection_reason"] = (
            "base LSTM selected without booster because no booster candidate satisfied the minimum trade coverage constraint"
        )

    _save_json(paths.report_path, train_report)
    _save_json(paths.latest_report_path, train_report)
    _save_json(paths.latest_file_report_path, train_report)
    if _is_legacy_default_pair(symbol, timeframe):
        _save_json(os.path.join(project_root, "training_report.json"), train_report)
        _save_json(os.path.join(project_root, "MQL5", "Files", "training_report.json"), train_report)
    _save_json(paths.state_path, {
        "run_id": paths.run_id,
        "run_dir": paths.run_dir,
        "resume_source": train_report["resume_source"],
        "resume_enabled": train_report["resumed_from_checkpoint"],
        "checkpoint_path": train_report["checkpoint_path"],
        "resume_checkpoint_path": train_report["resume_checkpoint_path"],
        "best_checkpoint_path": train_report["best_checkpoint_path"],
        "feature_contract_hash": train_report["feature_contract_hash"],
        "label_config_hash": train_report["label_config_hash"],
        "sequence_length": SEQ_LENGTH,
        "feature_count": len(feature_columns),
        "label_params": sweep_best_label_params,
        "class_histogram": train_report["class_histogram"],
        "epoch_completed": int(train_report["training_history"]["start_epoch"] + train_report["training_history"]["epochs_ran"]),
        "best_epoch": int(train_report["best_validation_epoch"]),
        "best_validation_score_tuple": train_report["best_validation_score_tuple"],
        "best_validation_loss": train_report["best_validation_loss"],
        "best_validation_metrics": train_report["best_validation_metrics"],
        "final_metrics": {
            "validation": combined_validation_metrics["trading_proxy"],
            "test": combined_test_metrics["trading_proxy"],
        },
        "booster": train_report["booster"],
    })

    stack_run_record = {
        "run_id": paths.run_id,
        "run_dir": paths.run_dir,
        "timestamp_utc": paths.run_id.split("_")[0],
        "record_kind": "stack_final",
        "feature_contract_hash": _feature_contract_hash(feature_columns),
        "label_config_hash": dp.build_label_config_hash(sweep_best_label_params),
        "sequence_length": SEQ_LENGTH,
        "training_schema_version": TRAINING_SCHEMA_VERSION,
        "model_family": MODEL_FAMILY,
        "label_params": sweep_best_label_params,
        "aux_horizons": sorted(int(h) for h in AUX_HORIZONS),
        "best_checkpoint_path": paths.best_path,
        "last_checkpoint_path": paths.last_path,
        "state_path": paths.state_path,
        "report_path": paths.report_path,
        "latest_report_path": paths.latest_report_path,
        "best_validation_score_tuple": train_report["best_validation_score_tuple"],
        "best_validation_metrics": train_report["best_validation_metrics"],
        "test_metrics": train_report["test_metrics"],
        "validation_metrics": train_report["validation_metrics"],
        "training_history": train_report["training_history"],
        "resume_source": train_report["resume_source"],
        "resume_checkpoint_path": train_report["resume_checkpoint_path"],
        "comparison_to_previous": train_report["comparison_to_previous"],
        "selection_reason": train_report["selection_reason"],
        "class_histogram": train_report["class_histogram"],
        "booster": train_report["booster"],
        "stack_metrics": train_report["final_stack_metrics"],
    }
    _append_jsonl(paths.index_path, stack_run_record)
    _save_json(paths.latest_pointer_path, {
        "run_id": paths.run_id,
        "run_dir": paths.run_dir,
        "report_path": paths.latest_report_path,
        "best_checkpoint_path": paths.best_path,
        "state_path": paths.state_path,
        "booster_onnx_path": train_report["booster"]["onnx_path"],
    })
    _update_best_registry(paths.best_registry_path, stack_run_record)
    if _is_legacy_default_pair(symbol, timeframe):
        _save_json(os.path.join(paths.history_dir, LATEST_POINTER_NAME), {
            "run_id": paths.run_id,
            "run_dir": paths.run_dir,
            "report_path": paths.latest_report_path,
            "best_checkpoint_path": paths.best_path,
            "state_path": paths.state_path,
            "booster_onnx_path": train_report["booster"]["onnx_path"],
        })
        _update_best_registry(os.path.join(paths.history_dir, BEST_REGISTRY_NAME), stack_run_record)

    final_report = {
        "symbol": symbol,
        "timeframe": timeframe,
        "csv_file": csv_file,
        "strict_eurusd_baseline_mode": strict_eurusd_baseline,
        "selection_reason": selection_reason,
        "selected_label_params": sweep_best_label_params,
        "selected_label_config_hash": dp.build_label_config_hash(sweep_best_label_params),
        "feature_contract": _feature_contract(feature_columns),
        "feature_contract_hash": _feature_contract_hash(feature_columns),
        "booster_feature_contract": {
            "feature_count": BOOSTER_INPUT_FEATURE_COUNT,
            "feature_columns": booster_feature_names,
        },
        "threshold_config": train_report["threshold_config"],
        "label_sweep_reports": [
            {
                "label_params": item["label_params"],
                "label_config_hash": item["label_config_hash"],
                "feature_contract_hash": item["feature_contract_hash"],
                "class_histogram": item["class_histogram"],
                "summary": item["summary"],
                "summary_tuple": item["summary_tuple"],
            }
            for item in sweep_reports
        ],
        "selected_walk_forward_summary": sweep_best_summary,
        "final_training_report": train_report,
        "stack_metrics": {
            "base_validation_metrics": base_validation_metrics,
            "base_test_metrics": base_test_metrics,
            "booster_validation_metrics": booster_stack["validation_metrics"] if booster_stack is not None else None,
            "combined_validation_metrics": combined_validation_metrics,
            "combined_test_metrics": combined_test_metrics,
        },
    }

    _save_json(os.path.join(paths.run_dir, "final_report.json"), final_report)
    _save_json(paths.latest_report_path, final_report)
    _save_json(paths.latest_file_report_path, final_report)
    if _is_legacy_default_pair(symbol, timeframe):
        _save_json(os.path.join(project_root, "training_report.json"), final_report)
        _save_json(os.path.join(project_root, "MQL5", "Files", "training_report.json"), final_report)
    print(f"Run report saved to {paths.report_path}", flush=True)
    print(f"Latest summary saved to {paths.latest_report_path}", flush=True)
    print(f"Model successfully exported to {paths.onnx_path}", flush=True)
    print(f"Copied model to {os.path.join(project_root, 'MQL5_EA', os.path.basename(paths.onnx_path))}", flush=True)
