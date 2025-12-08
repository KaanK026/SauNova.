"""
Utility scripts to generate supporting graphs and metrics for Saunova.

Run examples:
    python backend/analytics/reporting.py --csv backend/predictive_model/optimal_sauna_settings_with_height.csv
    python backend/analytics/reporting.py --session-json /path/to/session.json

Outputs are written under backend/analytics/outputs/ by default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backend.predictive_model.neural_network import SaunaRecommendationEngine

# Default paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = PROJECT_ROOT / "predictive_model" / "optimal_sauna_settings_with_height.csv"
DEFAULT_MODEL = PROJECT_ROOT / "predictive_model" / "sauna_recommendation_model.pth"
DEFAULT_SCALER = PROJECT_ROOT / "predictive_model" / "sauna_scaler.pkl"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


# ---------- helpers ----------

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_engine(model_path: Path = DEFAULT_MODEL, scaler_path: Path = DEFAULT_SCALER) -> SaunaRecommendationEngine | None:
    try:
        engine = SaunaRecommendationEngine(model_path=str(model_path), scaler_path=str(scaler_path))
        return engine
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[warn] Could not load model/scaler: {exc}")
        return None


def load_dataset(csv_path: Path = DEFAULT_CSV) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
    df = pd.read_csv(csv_path)
    return df


# ---------- plotting functions ----------

def plot_feature_distributions(df: pd.DataFrame, output_dir: Path) -> List[Path]:
    output_dir = ensure_dir(output_dir)
    feature_cols = [col for col in ["age", "BMI", "body_mass", "height"] if col in df.columns]
    paths: List[Path] = []
    for col in feature_cols:
        plt.figure(figsize=(6, 4))
        df[col].hist(bins=30, color="#1f77b4", edgecolor="white")
        plt.title(f"{col} distribution")
        plt.xlabel(col)
        plt.ylabel("count")
        out_path = output_dir / f"dist_{col}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        paths.append(out_path)
    return paths


def plot_goal_counts(df: pd.DataFrame, output_dir: Path) -> Path | None:
    if "goal" not in df.columns:
        return None
    output_dir = ensure_dir(output_dir)
    plt.figure(figsize=(7, 4))
    df["goal"].value_counts().plot(kind="bar", color="#ff7f0e", edgecolor="white")
    plt.title("Goal distribution")
    plt.xlabel("goal")
    plt.ylabel("count")
    out_path = output_dir / "goal_counts.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: Path) -> Path:
    output_dir = ensure_dir(output_dir)
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    plt.figure(figsize=(9, 7))
    plt.imshow(corr, cmap="coolwarm", interpolation="nearest")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.index)), corr.index)
    plt.colorbar(label="correlation")
    plt.title("Feature/target correlation matrix")
    plt.tight_layout()
    out_path = output_dir / "correlation_heatmap.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def _map_goal_to_frontend(goal_value: str, engine: SaunaRecommendationEngine) -> str:
    # goal values in CSV are already the canonical ones (e.g., stress_relief)
    if goal_value in engine.goal_mapping.values():
        return engine.reverse_goal_mapping.get(goal_value, "stress_reduction")
    # if goal_value already matches frontend keys, return as-is
    if goal_value in engine.goal_mapping:
        return goal_value
    return "stress_reduction"


def plot_predictions_vs_actuals(
    df: pd.DataFrame,
    engine: SaunaRecommendationEngine,
    output_dir: Path,
) -> Dict[str, Path]:
    required_cols = {"age", "height", "body_mass", "goal", "best_temp", "best_humidity", "best_session"}
    if not required_cols.issubset(df.columns):
        return {}

    output_dir = ensure_dir(output_dir)
    preds = {"temperature": [], "humidity": [], "session_length": []}
    actuals = {"temperature": [], "humidity": [], "session_length": []}

    for _, row in df.iterrows():
        try:
            goal_key = _map_goal_to_frontend(str(row["goal"]), engine)
            pred = engine.predict(
                age=float(row["age"]),
                gender="Other",
                height=float(row["height"]),
                weight=float(row["body_mass"]),
                selected_goals=[goal_key],
            )
            preds["temperature"].append(pred["temperature"])
            preds["humidity"].append(pred["humidity"])
            preds["session_length"].append(pred["session_length"])
            actuals["temperature"].append(row["best_temp"])
            actuals["humidity"].append(row["best_humidity"])
            actuals["session_length"].append(row["best_session"])
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[warn] prediction failed for row: {exc}")
            continue

    outputs: Dict[str, Path] = {}
    targets = [("temperature", "°C"), ("humidity", "%"), ("session_length", "minutes")]
    for key, unit in targets:
        if not preds[key]:
            continue
        plt.figure(figsize=(5, 5))
        plt.scatter(actuals[key], preds[key], alpha=0.6, color="#2ca02c", edgecolor="white")
        max_val = max(max(actuals[key]), max(preds[key])) + 5
        min_val = min(min(actuals[key]), min(preds[key])) - 5
        plt.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1)
        plt.xlabel(f"Actual {key} ({unit})")
        plt.ylabel(f"Predicted {key} ({unit})")
        plt.title(f"Predicted vs Actual: {key}")
        plt.tight_layout()
        out_path = output_dir / f"pred_vs_actual_{key}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        outputs[key] = out_path

    return outputs


def plot_learning_curves(log_csv: Path, output_dir: Path) -> Path | None:
    if not log_csv.exists():
        return None
    output_dir = ensure_dir(output_dir)
    df = pd.read_csv(log_csv)
    if not {"epoch", "train_loss", "val_loss"}.issubset(df.columns):
        return None
    plt.figure(figsize=(7, 4))
    plt.plot(df["epoch"], df["train_loss"], label="train")
    plt.plot(df["epoch"], df["val_loss"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Learning curves")
    plt.legend()
    plt.tight_layout()
    out_path = output_dir / "learning_curves.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_session_timeseries(
    timestamps: Sequence[float],
    temperatures: Sequence[float],
    humidities: Sequence[float],
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color_temp = "tab:red"
    color_hum = "tab:blue"

    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("temperature (°C)", color=color_temp)
    ax1.plot(timestamps, temperatures, color=color_temp, marker="o", linewidth=1.8)
    ax1.tick_params(axis="y", labelcolor=color_temp)

    ax2 = ax1.twinx()
    ax2.set_ylabel("humidity (%)", color=color_hum)
    ax2.plot(timestamps, humidities, color=color_hum, marker="s", linewidth=1.8)
    ax2.tick_params(axis="y", labelcolor=color_hum)

    plt.title("Sauna session telemetry")
    fig.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def plot_session_from_json(session_json: Path, output_dir: Path) -> Path | None:
    if not session_json.exists():
        return None
    with open(session_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    axis_data = payload.get("axis_data") or payload

    timestamps = axis_data.get("x_axis", {}).get("data") or axis_data.get("timestamps")
    temperatures = axis_data.get("y_axis_left", {}).get("data") or axis_data.get("temperatures")
    humidities = axis_data.get("y_axis_right", {}).get("data") or axis_data.get("humidities")

    if not (timestamps and temperatures and humidities):
        return None
    out_path = ensure_dir(output_dir) / "session_timeseries.png"
    return plot_session_timeseries(timestamps, temperatures, humidities, out_path)


# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(description="Generate supporting graphs for Saunova.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to sauna settings CSV.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Path to trained model (.pth).")
    parser.add_argument("--scaler", type=Path, default=DEFAULT_SCALER, help="Path to scaler (.pkl).")
    parser.add_argument("--log-csv", type=Path, help="Optional CSV with epoch,train_loss,val_loss for learning curves.")
    parser.add_argument("--session-json", type=Path, help="Optional session JSON payload with axis_data to plot.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory for graphs.")
    args = parser.parse_args()

    output_dir = ensure_dir(args.output)
    print(f"[info] outputs -> {output_dir}")

    # Dataset-driven plots
    try:
        df = load_dataset(args.csv)
        print(f"[info] loaded dataset with {len(df)} rows from {args.csv}")
    except Exception as exc:
        print(f"[warn] could not load dataset: {exc}")
        df = None

    engine = None
    if args.model.exists() and args.scaler.exists():
        engine = load_engine(args.model, args.scaler)
        if engine:
            print("[info] recommendation model loaded")
        else:
            print("[warn] recommendation model not loaded; prediction plots will be skipped")
    else:
        print("[warn] model/scaler files not found; skip prediction plots")

    if df is not None:
        plot_feature_distributions(df, output_dir)
        plot_goal_counts(df, output_dir)
        plot_correlation_heatmap(df, output_dir)
        if engine:
            plot_predictions_vs_actuals(df, engine, output_dir)

    if args.log_csv:
        plot_learning_curves(args.log_csv, output_dir)

    if args.session_json:
        plot_session_from_json(args.session_json, output_dir)

    print("[info] graph generation complete.")


if __name__ == "__main__":
    main()

