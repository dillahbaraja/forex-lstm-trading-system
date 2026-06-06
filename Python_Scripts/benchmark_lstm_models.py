import argparse
import csv
import datetime as _dt
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, Input, LSTM
from tensorflow.keras.models import Model, load_model

import data_preprocessing as dp
import train_model as tm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_ROOT = PROJECT_ROOT / "benchmark_runs"

SEED = 42
DEFAULT_EPOCHS = 140
DEFAULT_BATCH_SIZE = 64
DEFAULT_PATIENCE = 14


def _parse_args():
    parser = argparse.ArgumentParser(description="Run benchmark comparison for a specific symbol and timeframe.")
    parser.add_argument("--symbol", default=tm.DEFAULT_SYMBOL, help="Trading symbol, for example EURUSD, USDJPY, or EURJPY.")
    parser.add_argument("--timeframe", default=tm.DEFAULT_TIMEFRAME, help="Timeframe label used in file naming. Default: H1.")
    parser.add_argument("--csv", default=None, help="Optional explicit CSV path. Defaults to <SYMBOL>_<TIMEFRAME>_Data.csv in project root.")
    parser.add_argument(
        "--strict-eurusd-baseline",
        action="store_true",
        help="Force non-EURUSD pairs to reuse the final EURUSD label configuration for strict apple-to-apple comparison.",
    )
    return parser.parse_args()


def _pair_paths(symbol: str, timeframe: str):
    pair_slug = tm._pair_slug(symbol, timeframe)
    paths = {
        "data_csv": PROJECT_ROOT / tm._default_csv_name(symbol, timeframe),
        "training_report": PROJECT_ROOT / tm._artifact_name("training_report", pair_slug, "json"),
        "latest_run_pointer": PROJECT_ROOT / "training_runs" / tm._artifact_name("latest_run", pair_slug, "json"),
        "feature_contract": PROJECT_ROOT / "MQL5" / "Files" / tm._artifact_name("feature_contract", pair_slug, "json"),
        "benchmark_root": BENCHMARK_ROOT / pair_slug,
    }
    if tm._is_legacy_default_pair(symbol, timeframe):
        paths["legacy_training_report"] = PROJECT_ROOT / "training_report.json"
        paths["legacy_latest_run_pointer"] = PROJECT_ROOT / "training_runs" / "latest_run.json"
        paths["legacy_feature_contract"] = PROJECT_ROOT / "feature_contract.json"
    return paths


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        if math.isnan(value):
            return None
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _set_reproducibility(seed: int = SEED):
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def _now_utc_stamp() -> str:
    return _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_load_current_checkpoint(latest_run_pointer: Path) -> tuple[Model | None, Path | None, str]:
    if not latest_run_pointer.exists():
        return None, None, "latest_run_pointer_missing"

    latest_run = _read_json(latest_run_pointer)
    checkpoint_path = latest_run.get("best_checkpoint_path")
    if not checkpoint_path:
        return None, None, "best_checkpoint_path_missing"

    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        return None, checkpoint, "best_checkpoint_path_missing_on_disk"

    try:
        return load_model(str(checkpoint)), checkpoint, "loaded_best_checkpoint"
    except Exception as exc:
        return None, checkpoint, f"load_failed:{exc}"


def _current_label_params(training_report_path: Path) -> dict:
    if training_report_path.exists():
        report = _read_json(training_report_path)
        if "selected_label_params" in report:
            return dict(report["selected_label_params"])
        label_sweeps = report.get("label_sweep_reports") or []
        if label_sweeps:
            first = label_sweeps[0].get("label_params")
            if first:
                return dict(first)

    return {
        "future_horizon": 18,
        "tp_mult": 1.8,
        "sl_mult": 1.2,
        "min_move_atr": 0.35,
        "neutral_margin_atr": 0.05,
    }


def _strict_eurusd_label_params() -> dict:
    eurusd_report = PROJECT_ROOT / "training_report.json"
    if not eurusd_report.exists():
        raise FileNotFoundError("training_report.json not found. Strict EURUSD baseline mode requires the final EURUSD report.")
    report = _read_json(eurusd_report)
    label_params = report.get("selected_label_params")
    if not label_params:
        raise ValueError("training_report.json does not contain selected_label_params required for strict EURUSD baseline mode.")
    return dict(label_params)


def _current_feature_contract(feature_contract_path: Path) -> dict | None:
    if not feature_contract_path.exists():
        return None
    return _read_json(feature_contract_path)


def _build_paper_lstm_model(input_shape, num_classes=tm.NUM_CLASSES):
    inputs = Input(shape=input_shape, name="input_tensor")
    x = Dropout(0.15)(inputs)
    x = LSTM(96, return_sequences=False, name="paper_lstm")(x)
    x = Dropout(0.30)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.20)(x)
    outputs = Dense(num_classes, activation="softmax", name="main_output")(x)
    model = Model(inputs=inputs, outputs=outputs, name="paper_lstm_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _build_bilstm_model(input_shape, num_classes=tm.NUM_CLASSES):
    inputs = Input(shape=input_shape, name="input_tensor")
    x = Bidirectional(LSTM(64, return_sequences=True), name="bilstm_1")(inputs)
    x = Dropout(0.25)(x)
    x = Bidirectional(LSTM(32, return_sequences=False), name="bilstm_2")(x)
    x = Dropout(0.25)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.20)(x)
    outputs = Dense(num_classes, activation="softmax", name="main_output")(x)
    model = Model(inputs=inputs, outputs=outputs, name="bilstm_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _sequence_payload(features_df, targets, feature_columns, label_params, seq_length):
    scaler = dp.fit_scaler(features_df.iloc[: tm._final_split(len(features_df))[0]])
    scaled = dp.transform_features(features_df, scaler)
    X, y, end_indices = dp.create_sequences(scaled, targets, seq_length=seq_length, return_end_indices=True)
    train_end, val_end = tm._final_split(len(features_df))
    train_mask, val_mask, test_mask = tm._sequence_masks(end_indices, train_end, val_end)

    payload = {
        "scaler": scaler,
        "scaled_features": scaled,
        "X_train": X[train_mask],
        "y_train": y[train_mask],
        "X_val": X[val_mask],
        "y_val": y[val_mask],
        "X_test": X[test_mask],
        "y_test": y[test_mask],
        "end_indices": end_indices,
        "train_end": train_end,
        "val_end": val_end,
        "test_end": len(features_df),
        "feature_columns": list(feature_columns),
        "label_params": dict(label_params),
    }
    return payload


def _train_model(model_builder, input_shape, X_train, y_train, X_val, y_val):
    tf.keras.backend.clear_session()
    _set_reproducibility(SEED)
    model = model_builder(input_shape)
    sample_weight = compute_sample_weight(class_weight="balanced", y=np.asarray(y_train, dtype=int))
    callbacks = [
        EarlyStopping(monitor="val_loss", mode="min", patience=DEFAULT_PATIENCE, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", mode="min", factor=0.5, patience=max(4, DEFAULT_PATIENCE // 2), min_lr=1e-6, verbose=0),
    ]

    started = time.perf_counter()
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=DEFAULT_EPOCHS,
        batch_size=DEFAULT_BATCH_SIZE,
        callbacks=callbacks,
        sample_weight=sample_weight,
        verbose=0,
    )
    elapsed = time.perf_counter() - started
    return model, history, elapsed


def _predict_probabilities(model, X):
    probs = model.predict(X, verbose=0)
    return np.asarray(probs, dtype=np.float32)


def _evaluate_with_thresholds(y_true, y_prob, thresholds=None):
    if thresholds is None:
        tuned = tm._tune_gate_thresholds(
            y_true,
            y_prob,
            tm.evaluate_predictions,
            min_coverage=tm.MIN_COVERAGE_RATIO,
            min_trade_count=max(tm.MIN_TRADE_COUNT_FLOOR, int(len(y_true) * tm.MIN_TRADE_COUNT_FLOOR_RATIO)),
        )
        if tuned is None:
            raise RuntimeError("No valid threshold combination found for benchmark split.")
        thresholds = {
            "confidence_threshold": float(tuned["confidence_threshold"]),
            "min_confidence_edge": float(tuned["min_confidence_edge"]),
        }

    metrics = tm.evaluate_predictions(
        y_true,
        y_prob,
        confidence_threshold=float(thresholds["confidence_threshold"]),
        min_confidence_edge=float(thresholds["min_confidence_edge"]),
    )
    metrics["tuned_confidence_threshold"] = float(thresholds["confidence_threshold"])
    metrics["tuned_min_confidence_edge"] = float(thresholds["min_confidence_edge"])
    return metrics


def _classification_block(y_true, y_prob):
    y_pred = np.argmax(y_prob, axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1, 2],
            target_names=["DOWN", "SIDEWAYS", "UP"],
            zero_division=0,
            output_dict=True,
        ),
    }


def _package_metrics(model_name, source, model, train_time_sec, val_metrics, test_metrics, history=None):
    packaged = {
        "model_name": model_name,
        "source": source,
        "parameter_count": int(model.count_params()),
        "train_time_sec": float(train_time_sec),
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "validation_score_tuple": list(tm._metrics_tuple(val_metrics)),
        "test_score_tuple": list(tm._metrics_tuple(test_metrics)),
    }
    if history is not None:
        packaged["training_history"] = {
            "epochs_ran": int(len(history.history.get("loss", []))),
            "best_val_loss": float(min(history.history.get("val_loss", [math.inf]))),
            "final_loss": float(history.history.get("loss", [math.nan])[-1]),
        }
    return packaged


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_benchmark(symbol: str, timeframe: str, csv_override: str | None = None, strict_eurusd_baseline: bool = False):
    symbol = tm._normalize_symbol(symbol)
    timeframe = tm._normalize_timeframe(timeframe)
    pair_paths = _pair_paths(symbol, timeframe)
    data_csv = Path(csv_override).resolve() if csv_override else pair_paths["data_csv"]
    training_report_path = pair_paths["training_report"]
    latest_run_pointer = pair_paths["latest_run_pointer"]
    feature_contract_path = pair_paths["feature_contract"]

    legacy_training_report = pair_paths.get("legacy_training_report")
    legacy_latest_run_pointer = pair_paths.get("legacy_latest_run_pointer")
    legacy_feature_contract = pair_paths.get("legacy_feature_contract")
    if not training_report_path.exists() and legacy_training_report and legacy_training_report.exists():
        training_report_path = legacy_training_report
    if not latest_run_pointer.exists() and legacy_latest_run_pointer and legacy_latest_run_pointer.exists():
        latest_run_pointer = legacy_latest_run_pointer
    if not feature_contract_path.exists() and legacy_feature_contract and legacy_feature_contract.exists():
        feature_contract_path = legacy_feature_contract

    if not data_csv.exists():
        raise FileNotFoundError(f"{data_csv} not found.")

    _set_reproducibility(SEED)

    strict_eurusd_baseline = bool(strict_eurusd_baseline and symbol != tm.DEFAULT_SYMBOL)
    label_params = _strict_eurusd_label_params() if strict_eurusd_baseline else _current_label_params(training_report_path)
    feature_contract = _current_feature_contract(feature_contract_path)
    features_df, targets, feature_columns = dp.preprocess_data(str(data_csv), label_params=label_params)
    seq_length = int(tm.SEQ_LENGTH)

    if feature_contract:
        expected_columns = list(feature_contract.get("feature_columns") or [])
        expected_seq_length = int(feature_contract.get("sequence_length", seq_length))
        if expected_columns and list(feature_columns) != expected_columns:
            raise ValueError("Feature contract mismatch between preprocessing output and feature_contract.json.")
        if int(seq_length) != expected_seq_length:
            raise ValueError("Sequence length mismatch between benchmark code and feature_contract.json.")

    payload = _sequence_payload(features_df, targets, feature_columns, label_params, seq_length)

    if len(payload["X_train"]) == 0 or len(payload["X_val"]) == 0 or len(payload["X_test"]) == 0:
        raise RuntimeError("Benchmark split produced empty train/validation/test sequences.")

    benchmark_id = f"{_now_utc_stamp()}_{tm._pair_slug(symbol, timeframe)}_lstm_compare"
    output_dir = pair_paths["benchmark_root"] / benchmark_id
    output_dir.mkdir(parents=True, exist_ok=False)

    current_model, current_checkpoint, current_source = _safe_load_current_checkpoint(latest_run_pointer)
    if current_model is None:
        current_source = f"fallback_retrain:{current_source}"
        current_builder = lambda input_shape: tm.build_model(input_shape, num_classes=tm.NUM_CLASSES)
        current_model, current_history, current_train_time = _train_model(
            current_builder,
            (payload["X_train"].shape[1], payload["X_train"].shape[2]),
            payload["X_train"],
            payload["y_train"],
            payload["X_val"],
            payload["y_val"],
        )
    else:
        current_history = None
        current_train_time = 0.0

    model_specs = [
        {
            "key": "current_active",
            "display_name": "Current Active Architecture",
            "model": current_model,
            "source": current_source if current_checkpoint is None else str(current_checkpoint),
            "trained": current_history is not None,
        },
        {
            "key": "paper_lstm",
            "display_name": "Paper Reconstructed LSTM",
            "builder": _build_paper_lstm_model,
            "source": "reconstructed_from_paper_description",
        },
        {
            "key": "bilstm",
            "display_name": "BiLSTM Baseline",
            "builder": _build_bilstm_model,
            "source": "new_baseline_bilstm",
        },
    ]

    benchmark_models = {}
    rows_for_csv = []
    tuned_thresholds = {}

    for spec in model_specs:
        _set_reproducibility(SEED)
        if spec.get("trained"):
            model = spec["model"]
            history = current_history
            train_time_sec = current_train_time
        elif "model" in spec:
            model = spec["model"]
            history = None
            train_time_sec = 0.0
        else:
            model, history, train_time_sec = _train_model(
                spec["builder"],
                (payload["X_train"].shape[1], payload["X_train"].shape[2]),
                payload["X_train"],
                payload["y_train"],
                payload["X_val"],
                payload["y_val"],
            )

        val_prob = _predict_probabilities(model, payload["X_val"])
        val_metrics = _evaluate_with_thresholds(payload["y_val"], val_prob)
        tuned_thresholds[spec["key"]] = {
            "confidence_threshold": float(val_metrics["tuned_confidence_threshold"]),
            "min_confidence_edge": float(val_metrics["tuned_min_confidence_edge"]),
        }

        test_prob = _predict_probabilities(model, payload["X_test"])
        test_metrics = _evaluate_with_thresholds(payload["y_test"], test_prob, thresholds=tuned_thresholds[spec["key"]])

        benchmark_models[spec["key"]] = _package_metrics(
            spec["display_name"],
            spec["source"],
            model,
            train_time_sec,
            val_metrics,
            test_metrics,
            history=history,
        )

        rows_for_csv.append(
            {
                "model_key": spec["key"],
                "model_name": spec["display_name"],
                "validation_expectancy_r": benchmark_models[spec["key"]]["validation_metrics"]["trading_proxy"]["expectancy_r"],
                "validation_profit_factor": benchmark_models[spec["key"]]["validation_metrics"]["trading_proxy"]["profit_factor"],
                "validation_max_drawdown_r": benchmark_models[spec["key"]]["validation_metrics"]["trading_proxy"]["max_drawdown_r"],
                "validation_macro_f1": benchmark_models[spec["key"]]["validation_metrics"]["macro_f1"],
                "validation_weighted_f1": benchmark_models[spec["key"]]["validation_metrics"]["weighted_f1"],
                "validation_trade_count": benchmark_models[spec["key"]]["validation_metrics"]["trading_proxy"]["trade_count"],
                "test_expectancy_r": benchmark_models[spec["key"]]["test_metrics"]["trading_proxy"]["expectancy_r"],
                "test_profit_factor": benchmark_models[spec["key"]]["test_metrics"]["trading_proxy"]["profit_factor"],
                "test_max_drawdown_r": benchmark_models[spec["key"]]["test_metrics"]["trading_proxy"]["max_drawdown_r"],
                "test_macro_f1": benchmark_models[spec["key"]]["test_metrics"]["macro_f1"],
                "test_weighted_f1": benchmark_models[spec["key"]]["test_metrics"]["weighted_f1"],
                "test_trade_count": benchmark_models[spec["key"]]["test_metrics"]["trading_proxy"]["trade_count"],
            }
        )

    validation_rank = sorted(
        benchmark_models.items(),
        key=lambda item: tuple(item[1]["validation_score_tuple"]),
        reverse=True,
    )
    test_rank = sorted(
        benchmark_models.items(),
        key=lambda item: tuple(item[1]["test_score_tuple"]),
        reverse=True,
    )

    report = {
        "generated_at_utc": _now_utc_stamp(),
        "benchmark_id": benchmark_id,
        "symbol": symbol,
        "timeframe": timeframe,
        "strict_eurusd_baseline_mode": strict_eurusd_baseline,
        "data_csv": str(data_csv),
        "feature_contract": {
            "feature_count": len(feature_columns),
            "sequence_length": seq_length,
            "feature_columns": list(feature_columns),
        },
        "label_params": dict(label_params),
        "splits": {
            "train_end_row": int(payload["train_end"]),
            "val_end_row": int(payload["val_end"]),
            "test_end_row": int(payload["test_end"]),
            "train_sequence_count": int(len(payload["X_train"])),
            "validation_sequence_count": int(len(payload["X_val"])),
            "test_sequence_count": int(len(payload["X_test"])),
        },
        "models": benchmark_models,
        "ranking": {
            "validation": [item[0] for item in validation_rank],
            "test": [item[0] for item in test_rank],
            "validation_winner": validation_rank[0][0] if validation_rank else None,
            "test_winner": test_rank[0][0] if test_rank else None,
        },
        "thresholds": tuned_thresholds,
        "notes": [
            "Current active baseline is loaded from the latest Keras checkpoint when available.",
            "Paper baseline is reconstructed from the article description and mapped to the same 3-class target space.",
            "BiLSTM is trained on the same split, features, and label contract as the other baselines.",
        ],
    }

    report_path = output_dir / "benchmark_report.json"
    summary_csv_path = output_dir / "benchmark_summary.csv"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=_json_default)
    _write_csv(summary_csv_path, rows_for_csv)

    print(f"Benchmark written to: {output_dir}")
    print(f"Report: {report_path}")
    print(f"Summary CSV: {summary_csv_path}")
    print(f"Validation winner: {report['ranking']['validation_winner']}")
    print(f"Test winner: {report['ranking']['test_winner']}")


if __name__ == "__main__":
    args = _parse_args()
    run_benchmark(args.symbol, args.timeframe, args.csv, args.strict_eurusd_baseline)
