"""
nn_training_analysis.py
=======================
Utilities to load and compare neural-network training result JSON files.

Supports both epoch-level and step-level analysis (for step-based synchronization training).

Usage examples
--------------
    from nn_training_analysis import load_training_folder, load_from_paths, compare_runs, compare_speedups

    # Option A – load every JSON in a folder
    runs = load_training_folder("./results")

    # Option B – load specific files by path (as many as you need)
    runs = load_from_paths(
        "./results/loa.json",
        "./results/model_b.json",
        "/data/experiments/run_42.json",
    )

    # STANDARD COMPARISON: Compare all loaded runs (epoch-level)
    compare_runs(runs)

    # Compare with step-level granularity (for step-based training)
    compare_runs(runs, granularity="step")

    # Compare a named subset
    compare_runs(runs, keys=["loa", "model_b"])

    # Save the comparison to an HTML file
    compare_runs(runs, save_html="comparison.html")

    # SPEEDUP COMPARISON: Compare implementations relative to a baseline (epoch-level)
    # Speedup = baseline_time / implementation_time
    compare_speedups(runs, base_case="1worker", save_html="speedup_comparison.html")

    # Speedup comparison with step-level granularity
    compare_speedups(runs, base_case="1worker", granularity="step", save_html="speedup_steps.html")

    # Compare a subset with baseline
    compare_speedups(
        runs, 
        base_case="1worker",
        keys=["1worker", "2workers", "4workers"],
        save_html="speedup_subset.html"
    )
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

def runs_to_dataframe(runs: dict[str, dict], granularity: str = "epoch") -> pd.DataFrame:
    """
    Convert runs to a tidy DataFrame for plotting.
    
    Parameters
    ----------
    runs : dict[str, dict]
        Dictionary of loaded training run data.
    granularity : str
        Either "epoch" (default) for epoch-level metrics from historial_intervalo_*
        or "step" for step-level metrics from step_metrics.
    
    Returns
    -------
    pd.DataFrame
        Tidy dataframe with columns: model, epoch (or step_id), time_s, acc_train, loss
    """
    frames = []

    def safe_list(x):
        return x if isinstance(x, list) else []

    for name, data in runs.items():
        if granularity == "step":
            # ── STEP-LEVEL DATA ──────────────────────────────────────────────
            step_metrics = data.get("step_metrics", {})
            
            if not step_metrics:
                print(f"[WARNING] '{name}' has no step_metrics; skipping for step-level analysis")
                continue
            
            epoch_list = safe_list(step_metrics.get("step_ids", []))
            time   = safe_list(step_metrics.get("step_times", []))
            acc    = safe_list(step_metrics.get("step_accuracy", []))
            loss   = safe_list(step_metrics.get("step_loss", []))
            
            # Convert step_ids [[1,0], [1,1], ...] to step index 0, 1, 2, ...
            # (or keep epoch info if needed)
            step_indices = list(range(len(epoch_list))) if epoch_list else []
            
            print(f"[DEBUG] {name} (step-level) → "
                  f"steps={len(step_indices)}, time={len(time)}, "
                  f"acc={len(acc)}, loss={len(loss)}")
            
            if len(loss) == 0:
                min_len = min(len(step_indices), len(time), len(acc))
                loss = [None] * min_len
            else:
                min_len = min(len(step_indices), len(time), len(acc), len(loss))
            
            if min_len == 0:
                print(f"[WARNING] Skipping '{name}' (no usable step data)")
                continue
            
            frames.append(pd.DataFrame({
                "model"    : [name] * min_len,
                "epoch"    : step_indices[:min_len],  # step index
                "time_s"   : time[:min_len],
                "acc_train": acc[:min_len],
                "loss"     : loss[:min_len],
                "step_id"  : epoch_list[:min_len],    # [epoch, step] tuples
            }))
        
        else:
            # ── EPOCH-LEVEL DATA (DEFAULT) ───────────────────────────────────
            info = data.get("info_extra", {})

            epoch = safe_list(info.get("historial_intervalo_epochs"))
            time  = safe_list(info.get("historial_intervalo_times"))
            acc   = safe_list(info.get("historial_intervalo_acc_train"))
            loss  = safe_list(info.get("historial_intervalo_loss"))

            print(f"[DEBUG] {name} (epoch-level) → "
                  f"epoch={len(epoch)}, time={len(time)}, "
                  f"acc={len(acc)}, loss={len(loss)}")

            # 🔥 IGNORAR loss si está vacío
            if len(loss) == 0:
                min_len = min(len(epoch), len(time), len(acc))
                loss = [None] * min_len  # rellenar con NaN
            else:
                min_len = min(len(epoch), len(time), len(acc), len(loss))

            if min_len == 0:
                print(f"[WARNING] Skipping '{name}' (no usable epoch data)")
                continue

            frames.append(pd.DataFrame({
                "model"    : [name] * min_len,
                "epoch"    : epoch[:min_len],
                "time_s"   : time[:min_len],
                "acc_train": acc[:min_len],
                "loss"     : loss[:min_len],
            }))

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    print(f"[DEBUG] Final dataframe shape: {df.shape} (granularity={granularity})")

    return df


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


def compare_runs(
    runs: dict[str, dict],
    keys: Optional[list[str]] = None,
    save_html: Optional[str] = None,
    loose: bool = True,
    granularity: str = "epoch",
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
    granularity : str
        Either "epoch" (default) for epoch-level metrics or "step" for step-level metrics.

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
    df = runs_to_dataframe(selected, granularity=granularity)

    # ── shared colour encoding (same legend across all charts) ───────────────
    colour = alt.Color(
        "model:N",
        legend=alt.Legend(title="Model", orient="right"),
        scale=alt.Scale(scheme="category10"),
    )

    # ── Define labels based on granularity ──────────────────────────────────
    x_axis_label = "Epoch" if granularity == "epoch" else "Step"
    chart_title_prefix = "Epoch-level" if granularity == "epoch" else "Step-level"
    
    # ── shared colour encoding (same legend across all charts) ───────────────
    colour = alt.Color(
        "model:N",
        legend=alt.Legend(title="Model", orient="right"),
        scale=alt.Scale(scheme="category10"),
    )
    
    # ── tooltip definitions ──────────────────────────────────────────────────
    tt_epoch = [
        alt.Tooltip("model:N",     title="Model"),
        alt.Tooltip("epoch:Q",     title=x_axis_label),
        alt.Tooltip("acc_train:Q", title="Train Acc (%)", format=".2f"),
        alt.Tooltip("loss:Q",      title="Loss",          format=".4f"),
    ]
    tt_time = [
        alt.Tooltip("model:N",     title="Model"),
        alt.Tooltip("epoch:Q",     title=x_axis_label),
        alt.Tooltip("time_s:Q",    title="Time (s)",      format=".2f"),
        alt.Tooltip("acc_train:Q", title="Train Acc (%)", format=".2f"),
    ]

    # Each chart gets its own uniquely named pan/zoom selection to avoid
    # Altair's "deduplicated selection parameter" warning.
    zoom_a = alt.selection_interval(bind="scales", name="zoom_acc_epoch")
    zoom_b = alt.selection_interval(bind="scales", name="zoom_epoch_time")
    zoom_c = alt.selection_interval(bind="scales", name="zoom_loss_epoch")

    # ── Chart A – Epochs/Steps vs Time  (small, left) ───────────────────────────────
    chart_epoch_time = (
        alt.Chart(df)
        .mark_line(point=alt.OverlayMarkDef(size=20, opacity=0.5))
        .encode(
            y=alt.Y("time_s:Q", title="Elapsed Time (s)", scale=alt.Scale(nice=False)),
            x=alt.X("epoch:Q",  title=x_axis_label,       scale=alt.Scale(nice=False, zero=False)),
            color=colour,
            tooltip=tt_time,
        )
        .properties(
            title=alt.TitleParams(f"{chart_title_prefix} — {x_axis_label}s vs Time", fontSize=14),
            width=_W_MAIN,
            height=_H_MAIN,
        )
        .add_params(zoom_b)
    )

    # ── Chart B – Accuracy vs Epochs/Steps  (full-width, tall) ─────────────────────
    chart_acc_epoch = (
        alt.Chart(df)
        .mark_line(point=alt.OverlayMarkDef(size=20, opacity=0.5))
        .encode(
            x=alt.X("epoch:Q",     title=x_axis_label,            scale=alt.Scale(nice=False)),
            y=alt.Y("acc_train:Q", title="Training Accuracy (%)", scale=alt.Scale(zero=False)),
            color=colour,
            tooltip=tt_epoch,
        )
        .properties(
            title=alt.TitleParams(f"{chart_title_prefix} — Training Accuracy vs {x_axis_label}s", fontSize=14),
            width=_W_MAIN,
            height=_H_MAIN,
        )
        .add_params(zoom_a)
    )

    # ── Chart C – Loss vs Epochs/Steps  (small, right) ─────────────────────────────
    chart_loss_epoch = (
        alt.Chart(df)
        .mark_line(point=alt.OverlayMarkDef(size=20, opacity=0.5))
        .encode(
            x=alt.X("epoch:Q", title=x_axis_label, scale=alt.Scale(nice=False)),
            y=alt.Y("loss:Q",  title="Loss",      scale=alt.Scale(zero=False)),
            color=colour,
            tooltip=tt_epoch,
        )
        .properties(
            title=alt.TitleParams(f"{chart_title_prefix} — Loss vs {x_axis_label}s", fontSize=14),
            width=_W_MAIN,
            height=_H_MAIN,
        )
        .add_params(zoom_c)
    )

    combined = (
        alt.vconcat(chart_epoch_time , chart_acc_epoch, chart_loss_epoch)
        .resolve_scale(color="shared")
        .properties(
            title=alt.TitleParams(
                text=f"Training Comparison ({chart_title_prefix}) — {len(selected)} model(s)",
                fontSize=18,
            )
        )
    )

    if save_html:
        combined.save(save_html)
        print(f"Chart saved → {save_html}")

    return combined


# ──────────────────────────────────────────────────────────────────────────────
# 4. SPEEDUP COMPARISON MODULE
# ──────────────────────────────────────────────────────────────────────────────

def _normalize_base_case(base_case: str, runs: dict[str, dict]) -> str:
    """
    Normalize base_case to a model name. Handles both model names and file paths.
    
    If base_case is a model name already in runs, return it.
    If base_case is a file path, extract the stem and try to find it in runs.
    
    Parameters
    ----------
    base_case : str
        Either a model name or a file path.
    runs : dict[str, dict]
        Dictionary of loaded runs.
    
    Returns
    -------
    str
        The model name to use as baseline.
    
    Raises
    ------
    KeyError
        If the base_case (or its extracted stem) is not found in runs.
    """
    # First check if it's already a valid model name
    if base_case in runs:
        return base_case
    
    # Try extracting the stem from a potential file path
    extracted = Path(base_case).stem
    if extracted in runs:
        print(f"[INFO] Resolved base_case '{base_case}' → model '{extracted}'")
        return extracted
    
    # Not found in either format
    raise KeyError(
        f"Base case '{base_case}' not found. "
        f"Available models: {list(runs.keys())}"
    )


def speedups_to_dataframe(
    runs: dict[str, dict],
    base_case: str,
    granularity: str = "epoch",
) -> pd.DataFrame:
    """
    Calculate speedup values relative to a baseline implementation.

    Parameters
    ----------
    runs : dict[str, dict]
        Output of :func:`load_training_folder` or :func:`load_from_paths`.
    base_case : str
        Name of the baseline model to compare against. Can be a model name or file path.
    granularity : str
        Either "epoch" (default) for epoch-level metrics or "step" for step-level metrics.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: model, epoch (or step), speedup, time_s_baseline, time_s_impl
    """
    # Normalize base_case to handle both model names and file paths
    base_case = _normalize_base_case(base_case, runs)

    def safe_list(x):
        return x if isinstance(x, list) else []

    # Extract baseline data
    baseline_data = runs[base_case]
    
    if granularity == "step":
        # ── STEP-LEVEL DATA ──────────────────────────────────────────────
        baseline_metrics = baseline_data.get("step_metrics", {})
        if not baseline_metrics:
            raise ValueError(f"Base case '{base_case}' has no step_metrics data")
        
        baseline_time = safe_list(baseline_metrics.get("step_times", []))
        baseline_epoch = list(range(len(baseline_time))) if baseline_time else []
    else:
        # ── EPOCH-LEVEL DATA (DEFAULT) ───────────────────────────────────
        baseline_info = baseline_data.get("info_extra", {})
        baseline_time = safe_list(baseline_info.get("historial_intervalo_times"))
        baseline_epoch = safe_list(baseline_info.get("historial_intervalo_epochs"))

    if not baseline_time or not baseline_epoch:
        raise ValueError(f"Base case '{base_case}' has no training time data (granularity={granularity})")

    frames = []

    for name, data in runs.items():
        if name == base_case:
            # Baseline gets speedup = 1.0
            speedup_values = [1.0] * len(baseline_time)
            frames.append(pd.DataFrame({
                "model"           : [name] * len(baseline_time),
                "epoch"           : baseline_epoch[:len(baseline_time)],
                "speedup"         : speedup_values,
                "time_s_baseline" : baseline_time[:len(baseline_time)],
                "time_s_impl"     : baseline_time[:len(baseline_time)],
            }))
        else:
            # Other models: calculate speedup
            if granularity == "step":
                # ── STEP-LEVEL DATA ──────────────────────────────────────────────
                impl_metrics = data.get("step_metrics", {})
                if not impl_metrics:
                    print(f"[WARNING] Skipping '{name}' (no step_metrics data)")
                    continue
                
                impl_time = safe_list(impl_metrics.get("step_times", []))
                impl_epoch = list(range(len(impl_time))) if impl_time else []
            else:
                # ── EPOCH-LEVEL DATA (DEFAULT) ───────────────────────────────────
                info = data.get("info_extra", {})
                impl_time = safe_list(info.get("historial_intervalo_times"))
                impl_epoch = safe_list(info.get("historial_intervalo_epochs"))

            if not impl_time or not impl_epoch:
                print(f"[WARNING] Skipping '{name}' (no timing data for granularity={granularity})")
                continue

            # Align on common epoch count
            min_len = min(len(baseline_time), len(impl_time))
            if min_len == 0:
                print(f"[WARNING] Skipping '{name}' (no common data points)")
                continue

            # Calculate speedup: speedup = baseline_time / impl_time
            speedup_vals = [
                baseline_time[i] / impl_time[i] if impl_time[i] > 0 else 1.0
                for i in range(min_len)
            ]

            frames.append(pd.DataFrame({
                "model"           : [name] * min_len,
                "epoch"           : baseline_epoch[:min_len],
                "speedup"         : speedup_vals,
                "time_s_baseline" : baseline_time[:min_len],
                "time_s_impl"     : impl_time[:min_len],
            }))

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    print(f"[DEBUG] Speedup dataframe shape: {df.shape}")
    return df


def compare_speedups(
    runs: dict[str, dict],
    base_case: str,
    keys: Optional[list[str]] = None,
    save_html: Optional[str] = None,
    granularity: str = "epoch",
) -> alt.VConcatChart:
    """
    Create a speedup comparison visualization relative to a baseline implementation.

    Speedup is calculated as: speedup = time_baseline / time_implementation

    Parameters
    ----------
    runs : dict[str, dict]
        Output of :func:`load_training_folder` or :func:`load_from_paths`.
    base_case : str
        Name of the baseline model to use as reference (speedup = 1.0).
        Can be a model name (e.g., "oneWorker") or a file path 
        (e.g., "../stats/CIFAR_10/oneWorker.json").
    keys : list[str] | None
        Names of the runs to include. ``None`` → all runs.
    save_html : str | None
        Optional file path to export the chart as a standalone HTML file.
    granularity : str
        Either "epoch" (default) for epoch-level metrics or "step" for step-level metrics.

    Returns
    -------
    alt.VConcatChart
        Ready to display in a Jupyter notebook or save programmatically.

    Raises
    ------
    KeyError
        If base_case is not in runs or if any key in keys is not found.
    ValueError
        If base_case has no training time data.
    """
    # Normalize base_case to handle both model names and file paths
    base_case = _normalize_base_case(base_case, runs)

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

    if base_case not in selected:
        raise KeyError(f"Base case '{base_case}' not in selected runs")

    # ── build speedup dataframe ──────────────────────────────────────────────
    df = speedups_to_dataframe(selected, base_case, granularity=granularity)

    if df.empty:
        raise ValueError("No valid data to create speedup comparison")

    # ── Define labels based on granularity ──────────────────────────────────
    x_axis_label = "Epoch" if granularity == "epoch" else "Step"
    chart_title_prefix = "Epoch-level" if granularity == "epoch" else "Step-level"
    
    # ── shared colour encoding ───────────────────────────────────────────────
    colour = alt.Color(
        "model:N",
        legend=alt.Legend(title="Model", orient="right"),
        scale=alt.Scale(scheme="category10"),
    )

    # ── tooltip definitions ──────────────────────────────────────────────────
    tt_speedup = [
        alt.Tooltip("model:N",     title="Model"),
        alt.Tooltip("epoch:Q",     title=x_axis_label),
        alt.Tooltip("speedup:Q",   title="Speedup",          format=".2f"),
        alt.Tooltip("time_s_impl:Q", title="Implementation Time (s)", format=".2f"),
    ]

    # Selection for pan/zoom
    zoom_speedup = alt.selection_interval(bind="scales", name="zoom_speedup_epoch")

    # ── Chart A – Speedup vs Epochs/Steps (full width) ─────────────────────────────
    chart_speedup = (
        alt.Chart(df)
        .mark_line(point=alt.OverlayMarkDef(size=20, opacity=0.5))
        .encode(
            x=alt.X("epoch:Q",   title=x_axis_label, scale=alt.Scale(nice=False, zero=False)),
            y=alt.Y("speedup:Q", title="Speedup",    scale=alt.Scale(zero=False)),
            color=colour,
            tooltip=tt_speedup,
        )
        .properties(
            title=alt.TitleParams(f"{chart_title_prefix} — Speedup vs {x_axis_label}s", fontSize=14),
            width=_W_MAIN,
            height=_H_MAIN,
        )
        .add_params(zoom_speedup)
    )

    # ── Chart B – Time Reduction Percentage ───────────────────────────────────
    # Time reduction = (1 - impl_time / baseline_time) * 100 = (speedup - 1) / speedup * 100
    df["time_reduction_pct"] = ((df["speedup"] - 1) / df["speedup"]) * 100

    chart_time_reduction = (
        alt.Chart(df)
        .mark_line(point=alt.OverlayMarkDef(size=16, opacity=0.5))
        .encode(
            x=alt.X("epoch:Q",               title=x_axis_label, scale=alt.Scale(nice=False, zero=False)),
            y=alt.Y("time_reduction_pct:Q", title="Time Reduction (%)", scale=alt.Scale(zero=False)),
            color=colour,
            tooltip=[
                alt.Tooltip("model:N",             title="Model"),
                alt.Tooltip("epoch:Q",             title=x_axis_label),
                alt.Tooltip("time_reduction_pct:Q", title="Time Reduction (%)", format=".1f"),
            ],
        )
        .properties(
            title=alt.TitleParams(f"{chart_title_prefix} — Time Reduction %", fontSize=14),
            width=_W_MAIN,
            height=_H_MAIN,
        )
    )

    combined = (
        alt.vconcat(chart_speedup, chart_time_reduction)
        .resolve_scale(color="shared")
        .properties(
            title=alt.TitleParams(
                text=f"Speedup Comparison ({chart_title_prefix}, baseline: '{base_case}') — {len(selected)} model(s)",
                fontSize=18,
            )
        )
    )

    if save_html:
        combined.save(save_html)
        print(f"Chart saved → {save_html}")

    return combined

