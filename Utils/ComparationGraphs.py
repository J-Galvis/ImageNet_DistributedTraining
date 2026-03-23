"""
nn_training_analysis.py
=======================
Utilities to load and compare neural-network training result JSON files.

Usage examples
--------------
    from nn_training_analysis import load_training_folder, load_from_paths, compare_runs

    # Option A – load every JSON in a folder
    runs = load_training_folder("./results")

    # Option B – load specific files by path (as many as you need)
    runs = load_from_paths(
        "./results/loa.json",
        "./results/model_b.json",
        "/data/experiments/run_42.json",
    )

    # Compare all loaded runs
    compare_runs(runs)

    # Compare a named subset
    compare_runs(runs, keys=["loa", "model_b"])

    # Save the comparison to an HTML file
    compare_runs(runs, save_html="comparison.html")
"""

import json
from pathlib import Path
from typing import Optional

import altair as alt
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# 1. LOADER MODULE
# ──────────────────────────────────────────────────────────────────────────────

def _load_single(filepath: Path) -> tuple[str, dict]:
    """Read one JSON file and return ``(model_name, data)``."""
    with open(filepath, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    name = data.get("nombre_modelo") or filepath.stem
    return name, data


def _register(runs: dict, name: str, data: dict, filepath: Path) -> None:
    """Insert *data* into *runs*, deduplicating the key if necessary."""
    if name in runs:
        name = f"{name}__{filepath.stem}"
    runs[name] = data
    print(f"  ✔  Loaded '{name}'  ←  {filepath.name}")


def load_training_folder(folder: str) -> dict[str, dict]:
    """
    Scan *folder* for every ``*.json`` file and load each one.

    Parameters
    ----------
    folder : str
        Path to the directory that contains the JSON result files.

    Returns
    -------
    dict[str, dict]
        Keys are the model name (``nombre_modelo`` field inside the JSON,
        falling back to the filename stem when the field is absent).
        Values are the raw parsed dictionaries.

    Raises
    ------
    FileNotFoundError
        If *folder* does not exist.
    ValueError
        If no JSON files are found inside *folder*.
    """
    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder_path.resolve()}")

    json_files = sorted(folder_path.glob("*.json"))
    if not json_files:
        raise ValueError(f"No JSON files found in: {folder_path.resolve()}")

    runs: dict[str, dict] = {}
    for filepath in json_files:
        name, data = _load_single(filepath)
        _register(runs, name, data, filepath)

    print(f"\n{len(runs)} run(s) loaded from '{folder_path.resolve()}'.\n")
    return runs


def load_from_paths(*paths: str) -> dict[str, dict]:
    """
    Load an arbitrary number of JSON files given their individual paths.

    Parameters
    ----------
    *paths : str
        One or more file paths passed as positional arguments, e.g.::

            runs = load_from_paths(
                "./results/loa.json",
                "./results/model_b.json",
                "/data/run_42.json",
            )

    Returns
    -------
    dict[str, dict]
        Same structure as :func:`load_training_folder`:
        keys are model names, values are the raw parsed dictionaries.

    Raises
    ------
    ValueError
        If no paths are provided.
    FileNotFoundError
        If any of the provided paths does not point to an existing file.
    """
    if not paths:
        raise ValueError("Provide at least one file path.")

    runs: dict[str, dict] = {}
    for raw in paths:
        filepath = Path(raw)
        if not filepath.is_file():
            raise FileNotFoundError(f"File not found: {filepath.resolve()}")
        name, data = _load_single(filepath)
        _register(runs, name, data, filepath)

    print(f"\n{len(runs)} run(s) loaded.\n")
    return runs


# ──────────────────────────────────────────────────────────────────────────────
# 2. DATA HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def runs_to_dataframe(runs: dict[str, dict]) -> pd.DataFrame:
    """
    Flatten *runs* into a single tidy ``pd.DataFrame`` (one row per epoch
    per model).

    Columns: ``model``, ``epoch``, ``time_s``, ``acc_train``, ``loss``
    """
    frames = []
    for name, data in runs.items():
        info = data.get("info_extra", {})
        frames.append(pd.DataFrame({
            "model"    : name,
            "epoch"    : info.get("historial_intervalo_epochs",    []),
            "time_s"   : info.get("historial_intervalo_times",     []),
            "acc_train": info.get("historial_intervalo_acc_train", []),
            "loss"     : info.get("historial_intervalo_loss",      []),
        }))

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def runs_metadata(runs: dict[str, dict]) -> pd.DataFrame:
    """
    Return a summary ``pd.DataFrame`` with one row per model containing its
    high-level hyperparameters and results.
    """
    rows = []
    for name, data in runs.items():
        info = data.get("info_extra", {})
        arch = data.get("arquitectura", {})
        rows.append({
            "model"          : name,
            "test_acc (%)"   : data.get("precision_test"),
            "epochs"         : data.get("epocas"),
            "learning_rate"  : data.get("learning_rate"),
            "train_time (s)" : data.get("training_time_seconds"),
            "arch_hidden"    : arch.get("oculta"),
            "num_partitions" : info.get("num_particiones"),
            "architecture"   : info.get("architecture"),
        })
    return pd.DataFrame(rows).set_index("model")


# ──────────────────────────────────────────────────────────────────────────────
# 3. COMPARISON MODULE
# ──────────────────────────────────────────────────────────────────────────────

# Chart size constants – tweak these to taste
_W_MAIN  = 1000   # full-width chart
_H_MAIN  = 500
_W_SMALL =  490   # two small charts sit side-by-side inside the same 1000 px
_H_SMALL =  380


def compare_runs(
    runs: dict[str, dict],
    keys: Optional[list[str]] = None,
    save_html: Optional[str] = None,
) -> alt.VConcatChart:
    """
    Parameters
    ----------
    runs : dict[str, dict]
        Output of :func:`load_training_folder` or :func:`load_from_paths`.
    keys : list[str] | None
        Names of the runs to include.  ``None`` → all runs.
    save_html : str | None
        Optional file path to export the chart as a standalone HTML file.

    Returns
    -------
    alt.VConcatChart
        Ready to display in a Jupyter notebook or save programmatically.
    """
    # ── select subset ────────────────────────────────────────────────────────
    if keys:
        missing = [k for k in keys if k not in runs]
        if missing:
            raise KeyError(f"Keys not found in runs: {missing}")
        selected = {k: runs[k] for k in keys}
    else:
        selected = runs

    if not selected:
        raise ValueError("Need at least one run to compare.")

    # ── build tidy dataframe ─────────────────────────────────────────────────
    df = runs_to_dataframe(selected)

    # ── shared colour encoding (same legend across all charts) ───────────────
    colour = alt.Color(
        "model:N",
        legend=alt.Legend(title="Model", orient="right"),
        scale=alt.Scale(scheme="category10"),
    )

    # ── tooltip definitions ──────────────────────────────────────────────────
    tt_epoch = [
        alt.Tooltip("model:N",     title="Model"),
        alt.Tooltip("epoch:Q",     title="Epoch"),
        alt.Tooltip("acc_train:Q", title="Train Acc (%)", format=".2f"),
        alt.Tooltip("loss:Q",      title="Loss",          format=".4f"),
    ]
    tt_time = [
        alt.Tooltip("model:N",     title="Model"),
        alt.Tooltip("epoch:Q",     title="Epoch"),
        alt.Tooltip("time_s:Q",    title="Time (s)",      format=".2f"),
        alt.Tooltip("acc_train:Q", title="Train Acc (%)", format=".2f"),
    ]

    # Each chart gets its own uniquely named pan/zoom selection to avoid
    # Altair's "deduplicated selection parameter" warning.
    zoom_a = alt.selection_interval(bind="scales", name="zoom_acc_epoch")
    zoom_b = alt.selection_interval(bind="scales", name="zoom_epoch_time")
    zoom_c = alt.selection_interval(bind="scales", name="zoom_loss_epoch")

        # ── Chart A – Epochs vs Time  (small, left) ───────────────────────────────
    chart_epoch_time = (
        alt.Chart(df)
        .mark_line(point=alt.OverlayMarkDef(size=20, opacity=0.5))
        .encode(
            y=alt.Y("time_s:Q", title="Elapsed Time (s)", scale=alt.Scale(nice=False)),
            x=alt.X("epoch:Q",  title="Epoch",            scale=alt.Scale(nice=False, zero=False)),
            color=colour,
            tooltip=tt_time,
        )
        .properties(
            title=alt.TitleParams("Training Epochs vs Time", fontSize=14),
            width=_W_MAIN,
            height=_H_MAIN,
        )
        .add_params(zoom_b)
    )

    # ── Chart B – Accuracy vs Epochs  (full-width, tall) ─────────────────────
    chart_acc_epoch = (
        alt.Chart(df)
        .mark_line(point=alt.OverlayMarkDef(size=20, opacity=0.5))
        .encode(
            x=alt.X("epoch:Q",     title="Epoch",                 scale=alt.Scale(nice=False)),
            y=alt.Y("acc_train:Q", title="Training Accuracy (%)", scale=alt.Scale(zero=False)),
            color=colour,
            tooltip=tt_epoch,
        )
        .properties(
            title=alt.TitleParams("Training Accuracy vs Epochs", fontSize=14),
            width=_W_SMALL,
            height=_H_SMALL,
        )
        .add_params(zoom_a)
    )

    # ── Chart C – Loss vs Epochs  (small, right) ─────────────────────────────
    chart_loss_epoch = (
        alt.Chart(df)
        .mark_line(point=alt.OverlayMarkDef(size=20, opacity=0.5))
        .encode(
            x=alt.X("epoch:Q", title="Epoch", scale=alt.Scale(nice=False)),
            y=alt.Y("loss:Q",  title="Loss",  scale=alt.Scale(zero=False)),
            color=colour,
            tooltip=tt_epoch,
        )
        .properties(
            title=alt.TitleParams("Loss vs Epochs", fontSize=14),
            width=_W_SMALL,
            height=_H_SMALL,
        )
        .add_params(zoom_c)
    )

    # ── Compose layout ────────────────────────────────────────────────────────
    #   Row 1 → Chart A  (full width)
    #   Row 2 → Chart B | Chart C  (side by side, independent Y axes)
    bottom_row = (
        alt.hconcat(chart_acc_epoch, chart_loss_epoch)
        .resolve_scale(color="shared", y="independent")
    )

    combined = (
        alt.vconcat(chart_epoch_time , bottom_row)
        .resolve_scale(color="shared")
        .properties(
            title=alt.TitleParams(
                text=f"Training Comparison — {len(selected)} model(s)",
                fontSize=18,
            )
        )
    )

    if save_html:
        combined.save(save_html)
        print(f"Chart saved → {save_html}")

    return combined

