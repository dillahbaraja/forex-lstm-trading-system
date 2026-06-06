import csv
import json
import os
import hashlib

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

FEATURE_COLUMNS = [
    "Open", "High", "Low", "Close", "Volume",
    "RSI", "ATR", "MACD_Main", "MACD_Signal",
    "Bands_Upper", "Bands_Lower",
    "Stoch_Main", "Stoch_Signal",
    "ADX_Main", "ADX_PlusDI", "ADX_MinusDI",
    "RSI_H4", "MA_H4", "RSI_M15", "ATR_M15",
    "Hour_Sin", "Hour_Cos", "DayOfWeek_Sin", "DayOfWeek_Cos",
    "Daily_Trend_Proxy",
]

TARGET_MAP = {
    0: "DOWN",
    1: "SIDEWAYS",
    2: "UP",
}


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if "Time" in df.columns:
        df["Time"] = pd.to_datetime(df["Time"])
        df.set_index("Time", inplace=True)
    return df


def build_label_config_hash(label_params: dict) -> str:
    normalized = json.dumps(label_params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _build_triple_barrier_targets(
    df: pd.DataFrame,
    future_horizon: int = 24,
    tp_mult: float = 2.0,
    sl_mult: float = 1.5,
    min_move_atr: float = 0.0,
    neutral_margin_atr: float = 0.05,
) -> np.ndarray:
    close_arr = df["Close"].values
    high_arr = df["High"].values
    low_arr = df["Low"].values
    atr_arr = df["ATR"].values

    targets_list = np.ones(len(df), dtype=int)

    for i in range(len(df) - future_horizon):
        c = close_arr[i]
        a = atr_arr[i]
        if a <= 0:
            targets_list[i] = 1
            continue

        tp_price = c + (tp_mult * a)
        sl_price = c - (sl_mult * a)
        max_up_move = 0.0
        max_down_move = 0.0
        up_touch_idx = None
        down_touch_idx = None

        for k in range(1, future_horizon + 1):
            bar_high = high_arr[i + k]
            bar_low = low_arr[i + k]

            max_up_move = max(max_up_move, (bar_high - c) / a)
            max_down_move = max(max_down_move, (c - bar_low) / a)

            if up_touch_idx is None and bar_high >= tp_price:
                up_touch_idx = k
            if down_touch_idx is None and bar_low <= sl_price:
                down_touch_idx = k

            if up_touch_idx is not None and down_touch_idx is not None:
                break

        if up_touch_idx is not None and down_touch_idx is not None:
            if up_touch_idx == down_touch_idx:
                targets_list[i] = 1
            else:
                targets_list[i] = 2 if up_touch_idx < down_touch_idx else 0
            continue

        if up_touch_idx is not None:
            if (max_up_move - max_down_move) < neutral_margin_atr:
                targets_list[i] = 1
            else:
                targets_list[i] = 2
            continue

        if down_touch_idx is not None:
            if (max_down_move - max_up_move) < neutral_margin_atr:
                targets_list[i] = 1
            else:
                targets_list[i] = 0
            continue

        if max(max_up_move, max_down_move) < min_move_atr:
            targets_list[i] = 1
        elif abs(max_up_move - max_down_move) < neutral_margin_atr:
            targets_list[i] = 1
        elif max_up_move > max_down_move and max_up_move >= min_move_atr:
            targets_list[i] = 2
        elif max_down_move > max_up_move and max_down_move >= min_move_atr:
            targets_list[i] = 0
        else:
            targets_list[i] = 1

    return targets_list


def build_multi_horizon_targets(
    df: pd.DataFrame,
    main_label_params: dict,
    aux_horizons: tuple[int, ...] | list[int] | None = None,
) -> dict[int, np.ndarray]:
    if aux_horizons is None:
        aux_horizons = ()

    horizons = sorted(set([int(main_label_params.get("future_horizon", 24))] + [int(h) for h in aux_horizons]))
    targets = {}
    for horizon in horizons:
        horizon_params = dict(main_label_params)
        horizon_params["future_horizon"] = horizon
        targets[horizon] = _build_triple_barrier_targets(
            df,
            future_horizon=int(horizon_params.get("future_horizon", 24)),
            tp_mult=float(horizon_params.get("tp_mult", 2.0)),
            sl_mult=float(horizon_params.get("sl_mult", 1.5)),
            min_move_atr=float(horizon_params.get("min_move_atr", 0.5)),
            neutral_margin_atr=float(horizon_params.get("neutral_margin_atr", 0.0)),
        )
    return targets


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Hour" in df.columns:
        df["Hour_Sin"] = np.sin(2 * np.pi * df["Hour"] / 24.0)
        df["Hour_Cos"] = np.cos(2 * np.pi * df["Hour"] / 24.0)
        df.drop(columns=["Hour"], inplace=True)

    if "DayOfWeek" in df.columns:
        df["DayOfWeek_Sin"] = np.sin(2 * np.pi * df["DayOfWeek"] / 7.0)
        df["DayOfWeek_Cos"] = np.cos(2 * np.pi * df["DayOfWeek"] / 7.0)
        df.drop(columns=["DayOfWeek"], inplace=True)

    df["Daily_Trend_Proxy"] = (df["Close"] - df["Close"].rolling(window=24).mean()) / df["Close"]

    if "Volume" in df.columns:
        df["Volume"] = np.log1p(df["Volume"])

    df["Bands_Upper"] = (df["Bands_Upper"] - df["Close"]) / df["Close"]
    df["Bands_Lower"] = (df["Bands_Lower"] - df["Close"]) / df["Close"]

    if "MA_H4" in df.columns:
        df["MA_H4"] = (df["Close"] - df["MA_H4"]) / df["Close"]

    df["Open"] = df["Open"].pct_change()
    df["High"] = df["High"].pct_change()
    df["Low"] = df["Low"].pct_change()
    df["Close"] = df["Close"].pct_change()

    df.dropna(inplace=True)
    missing_columns = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing expected feature columns: {missing_columns}")

    return df


def preprocess_data(csv_path: str, label_params: dict | None = None, aux_horizons: tuple[int, ...] | list[int] | None = None, return_aux_targets: bool = False):
    """
    Load the exported CSV, build triple-barrier labels, and engineer raw features.
    Returns an unscaled feature frame, targets, and the feature contract.
    """
    if label_params is None:
        label_params = {
            "future_horizon": 24,
            "tp_mult": 2.0,
            "sl_mult": 1.5,
            "min_move_atr": 0.5,
            "neutral_margin_atr": 0.08,
        }

    if aux_horizons is None:
        aux_horizons = (6, 12, 24, 48)

    future_horizon = int(label_params.get("future_horizon", 24))
    tp_mult = float(label_params.get("tp_mult", 2.0))
    sl_mult = float(label_params.get("sl_mult", 1.5))
    min_move_atr = float(label_params.get("min_move_atr", 0.5))
    neutral_margin_atr = float(label_params.get("neutral_margin_atr", 0.08))

    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    df = _ensure_datetime_index(df)
    print(f"Data loaded. Shape: {df.shape}")

    target_horizons = sorted(set([future_horizon] + [int(h) for h in aux_horizons]))
    target_columns = {}
    for horizon in target_horizons:
        target_columns[horizon] = _build_triple_barrier_targets(
            df,
            future_horizon=int(horizon),
            tp_mult=tp_mult,
            sl_mult=sl_mult,
            min_move_atr=min_move_atr,
            neutral_margin_atr=neutral_margin_atr,
        )

    max_horizon = max(target_horizons)
    df = df.iloc[:-max_horizon].copy()
    for horizon, targets_arr in target_columns.items():
        df[f"Target_{horizon}"] = targets_arr[: len(df)]
    df = _engineer_features(df)

    targets = df[f"Target_{future_horizon}"].astype(int).values
    features_df = df[FEATURE_COLUMNS].copy()
    aux_targets = {
        horizon: df[f"Target_{horizon}"].astype(int).values
        for horizon in target_horizons
        if horizon != future_horizon
    }

    print(f"Features engineered. New shape: {features_df.shape}")
    class_counts = pd.Series(targets).value_counts().sort_index()
    print("Target distribution:", {TARGET_MAP.get(int(k), str(k)): int(v) for k, v in class_counts.items()})

    if return_aux_targets:
        return features_df, targets, FEATURE_COLUMNS.copy(), aux_targets
    return features_df, targets, FEATURE_COLUMNS.copy()


def fit_scaler(train_features: pd.DataFrame, output_path: str | None = None, file_copy_path: str | None = None):
    scaler = RobustScaler()
    scaler.fit(train_features)
    if output_path is not None or file_copy_path is not None:
        save_scaler_params(scaler, output_path=output_path, file_copy_path=file_copy_path)
    return scaler


def save_scaler_params(scaler, output_path: str | None = None, file_copy_path: str | None = None):
    if output_path is None:
        output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scaler_params.csv"))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([float(v) for v in scaler.center_])
        writer.writerow([float(v) for v in scaler.scale_])

    if file_copy_path:
        os.makedirs(os.path.dirname(file_copy_path), exist_ok=True)
        with open(file_copy_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([float(v) for v in scaler.center_])
            writer.writerow([float(v) for v in scaler.scale_])

    print(f"Scaler parameters saved to {output_path}")
    if file_copy_path:
        print(f"Scaler parameters copied to {file_copy_path}")


def save_scaler_include(scaler, output_path: str | None = None, file_copy_path: str | None = None):
    if output_path is None:
        output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "GeneratedScaler.mqh"))

    def _format_array(name: str, values: np.ndarray) -> str:
        values_str = ", ".join(f"{float(v):.17g}" for v in values)
        return f"double {name}[{len(values)}] = {{{values_str}}};\n"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    content = [
        "// Auto-generated by train_model.py. Do not edit manually.\n",
        "// This file embeds scaler parameters into the EA build.\n\n",
        _format_array("ScalerCenter", np.asarray(scaler.center_, dtype=float)),
        _format_array("ScalerScale", np.asarray(scaler.scale_, dtype=float)),
    ]
    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(content)

    if file_copy_path:
        os.makedirs(os.path.dirname(file_copy_path), exist_ok=True)
        with open(file_copy_path, "w", encoding="utf-8", newline="\n") as f:
            f.writelines(content)

    print(f"Scaler include saved to {output_path}")
    if file_copy_path:
        print(f"Scaler include copied to {file_copy_path}")


def save_feature_contract(feature_columns, output_path: str | None = None, file_copy_path: str | None = None):
    contract = {
        "feature_count": len(feature_columns),
        "sequence_length": 24,
        "feature_columns": list(feature_columns),
        "target_map": TARGET_MAP,
    }

    if output_path is None:
        output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "feature_contract.json"))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(contract, f, indent=2)

    if file_copy_path:
        os.makedirs(os.path.dirname(file_copy_path), exist_ok=True)
        with open(file_copy_path, "w", encoding="utf-8") as f:
            json.dump(contract, f, indent=2)

    print(f"Feature contract saved to {output_path}")
    if file_copy_path:
        print(f"Feature contract copied to {file_copy_path}")


def save_label_config(label_params: dict, output_path: str | None = None, file_copy_path: str | None = None):
    payload = {
        "label_params": label_params,
        "label_config_hash": build_label_config_hash(label_params),
        "target_map": TARGET_MAP,
        "aux_horizons": label_params.get("aux_horizons"),
    }

    if output_path is None:
        output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "label_config.json"))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    if file_copy_path:
        os.makedirs(os.path.dirname(file_copy_path), exist_ok=True)
        with open(file_copy_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    print(f"Label config saved to {output_path}")
    if file_copy_path:
        print(f"Label config copied to {file_copy_path}")


def transform_features(features: pd.DataFrame, scaler: RobustScaler) -> np.ndarray:
    return scaler.transform(features)


def create_sequences(features, targets, seq_length: int = 24, return_end_indices: bool = False):
    X, y, end_indices = [], [], []
    for i in range(len(features) - seq_length):
        end_idx = i + seq_length
        X.append(features[i:end_idx])
        y.append(targets[end_idx])
        end_indices.append(end_idx)

    X = np.array(X)
    y = np.array(y)
    end_indices = np.array(end_indices)

    if return_end_indices:
        return X, y, end_indices
    return X, y


if __name__ == "__main__":
    pass
