"""
HICC – Recall & Label-Flip Explorer (Streamlit)

An interactive dashboard to analyze LLM moderation results on the HICC dataset
from a single CSV file. The app has two tabs:

• Label-Flip Analysis
    Shows how binary verdicts change as more conversational context is added
    (“g-n-o-f” = ground truth – no_context – with_original – with_full_context).
    Useful for spotting brittle behavior and context sensitivity.

• Recall Analysis
    Computes recall across models and context conditions, both globally and
    in bins over an aggregate difficulty score. Includes “classic” recall
    (per task), cross-prompt recall (tox on hate, hate on tox), and
    union-of-prompts variants.

Input
-----
A CSV (default path: CSV_PATH) with at least:
    - annotation (string with labels; e.g. contains INTERVENTION_HATE / _TOXIC)
    - is_annotated_text (0/1 or truthy string)
    - aggregate_avg (float; used for binning)
    - *_verdict columns for each (task, model, context), e.g.:
        hate_openai_no_context_verdict
        hate_openai_with_original_verdict
        hate_openai_with_full_context_verdict
        toxicity_xai_no_context_verdict
        ...

Filtering
---------
Rows are restricted to:
    is_annotated_text == 1  AND  hate_openai_no_context_verdict is not null

UI Controls
-----------
- CSV path input or file upload (sidebar)
- Number of bins for recall vs. aggregate_avg (default=10, as used in the results shown in the paper)
- Metric selector (classic, cross-prompt, union)
- Label-flip toggles (task, counts vs. %, include ground truth)

Outputs
-------
- On-screen tables (summary & per-bin)
- Per-model bar charts and delta line charts (downloadable PNGs)
- Combined figure of all three models (downloadable PNG)

Run
---
    streamlit run evaluate_recall.py

Requirements
------------
    pip install streamlit pandas matplotlib numpy
"""

# ──────────────────────────────  IMPORTS  ──────────────────────────────

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import io
import base64
from pathlib import Path

# ──────────────────────────────  CONFIG  ──────────────────────────────

CSV_PATH = "dataset/hicc_dataset.csv"
MODELS = ["openai", "xai", "claude"]
MODEL_NAMES = {
    "openai": "GPT-4o-mini",
    "xai": "Grok-3-mini",
    "claude": "Claude 3 Haiku",
}
CONTEXTS = ["no_context", "with_original", "with_full_context"]
GROUND_TRUTH_HATE = "INTERVENTION_HATE"
GROUND_TRUTH_TOXIC = "INTERVENTION_TOXIC"
BAR_COLORS = {  # separate palette for bar charts
    "no_context": "#1f77b4",  # blue
    "with_original": "#ff7f0e",  # orange
    "with_full_context": "#2ca02c",  # green
}
LINE_COLORS = {
    "with_original - no_context": "#d62728",  # red
    "with_full_context - no_context": "#9467bd",  # purple
}

# ──────────────────────────────  HELPER FUNCTIONS  ──────────────────────────────


def as_bool(val) -> bool:
    """Coerce diverse truthy representations into a boolean."""
    if pd.isna(val):
        return False
    if isinstance(val, (int, float)):
        try:
            return int(val) != 0
        except Exception:
            return False
    s = str(val).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


def coerce_verdict_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all *_verdict columns are numeric (0/1 or NaN)."""
    for task in ("hate", "toxicity"):
        for model in MODELS:
            for context in CONTEXTS:
                column = f"{task}_{model}_{context}_verdict"
                if column in df.columns:
                    df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def load_evaluated_entries_from_csv(path_or_buffer) -> pd.DataFrame:
    df = pd.read_csv(path_or_buffer)
    # essential coercions
    if "is_annotated_text" in df.columns:
        df["is_annotated_text"] = df["is_annotated_text"].apply(as_bool)
    else:
        df["is_annotated_text"] = False

    df["annotation"] = df.get("annotation", "").fillna("").astype(str)
    if "aggregate_avg" in df.columns:
        df["aggregate_avg"] = pd.to_numeric(df["aggregate_avg"], errors="coerce")

    df = coerce_verdict_columns(df)

    # mirror the previous DB filter:
    # WHERE is_annotated_text = 1 AND hate_openai_no_context_verdict IS NOT NULL
    if "hate_openai_no_context_verdict" in df.columns:
        df = df[df["is_annotated_text"] & df["hate_openai_no_context_verdict"].notna()]
    else:
        # if column missing, keep only annotated rows
        df = df[df["is_annotated_text"]]

    return df


def safe_divide(tp: int, total: int) -> float:
    return 0 if total == 0 else tp / total


def verdict_equals_one(row, col):
    return (col in row) and (row[col] == 1)


def bin_midpoint(interval):
    return (interval.left + interval.right) / 2


# LABEL-FLIP LOGIC


def derive_ground_truth(row: pd.Series, task: str) -> int:
    annotation: str | None = row.get("annotation", "")
    if task == "hate":
        return int(bool(annotation and GROUND_TRUTH_HATE in annotation))
    if task == "toxicity":
        return int(bool(annotation and GROUND_TRUTH_TOXIC in annotation))
    raise ValueError(task)


def compute_label_flip_patterns(
    df: pd.DataFrame,
    model: str,
    task: str,
    include_ground_truth: bool = True,
) -> pd.DataFrame:
    """Return DF with columns [pattern, count, pct].
    If *include_gt* is False the pattern is reduced to "n-o-f" (8 combos)."""

    verdict_columns = [f"{task}_{model}_{ctx}_verdict" for ctx in CONTEXTS]
    combinations: defaultdict[str, int] = defaultdict(int)

    for _, row in df.iterrows():
        if any(pd.isna(row.get(c)) for c in verdict_columns):
            continue
        no_context, original_context, full_context = (
            int(row[v]) for v in verdict_columns
        )
        if include_ground_truth:
            ground_truth = derive_ground_truth(row, task)
            pattern = f"{ground_truth}-{no_context}-{original_context}-{full_context}"
        else:
            pattern = f"{no_context}-{original_context}-{full_context}"
        combinations[pattern] += 1

    if not combinations:
        return pd.DataFrame(columns=["pattern", "count", "pct"])

    total = sum(combinations.values())
    if include_ground_truth:
        ordered = [
            f"{g}-{a}-{b}-{c}"
            for g in (0, 1)
            for a in (0, 1)
            for b in (0, 1)
            for c in (0, 1)
        ]
    else:
        ordered = [f"{a}-{b}-{c}" for a in (0, 1) for b in (0, 1) for c in (0, 1)]

    return pd.DataFrame(
        [
            {
                "pattern": p,
                "count": combinations.get(p, 0),
                "pct": combinations.get(p, 0) / total,
            }
            for p in ordered
        ]
    )


# ──────────────────────────────  STREAMLIT UI  ──────────────────────────────

st.set_page_config(layout="wide")
st.title("Analysis of LLM-Judges on HICC Dataset (CSV)")

# Global sidebar controls (apply to both tabs)
with st.sidebar:
    st.header("Data & Global Controls")

    # Choose CSV via path or upload
    default_path = st.text_input("CSV path", value=CSV_PATH)
    uploaded = st.file_uploader("…or upload CSV", type=["csv"])

    num_bins = st.slider("Number of bins (recall)", 1, 20, 10)

# ──────────────────────────────  LOAD DATA  ──────────────────────────────

try:
    if uploaded is not None:
        df = load_evaluated_entries_from_csv(uploaded)
    else:
        csv_file = Path(default_path)
        if not csv_file.exists():
            st.error(f"CSV not found: {csv_file.resolve()}")
            st.stop()
        df = load_evaluated_entries_from_csv(csv_file)
except Exception as e:
    st.error(f"Failed to load CSV: {e}")
    st.stop()

if df.empty:
    st.error("No evaluated entries found in the CSV after filtering.")
    st.stop()

# annotation masks (safer .str.contains with na=False)
is_toxic_speech = df["annotation"].str.contains(GROUND_TRUTH_TOXIC, na=False)
is_hate_speech = df["annotation"].str.contains(GROUND_TRUTH_HATE, na=False)

# ──────────────────────────────  TABS  ──────────────────────────────

tab_recall, tab_flip = st.tabs(["Recall Analysis", "Label Flip Analysis"])

with tab_flip:
    st.subheader(
        "Label-Flip Patterns",
        help="g = ground truth, n = no context (target text only), o = plus original comment (root + target), f = with full context (root + prior direct replies + target)",
    )

    # Controls
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
    with col_ctrl1:
        task_choice = st.radio("Task", ["hate", "toxicity"], horizontal=True)
    with col_ctrl2:
        display_mode = st.radio("Display", ["Counts", "Percentages"], horizontal=True)
    with col_ctrl3:
        include_ground_truth = st.checkbox("Include ground-truth digit", value=True)

    for model in MODELS:
        # st.markdown(f"### {model.upper()}")
        model_name = MODEL_NAMES.get(model, "unknown")
        st.markdown(f"### {model_name}")
        df_pat = compute_label_flip_patterns(
            df, model, task_choice, include_ground_truth
        )
        if df_pat.empty:
            st.info("No complete rows for this model/task.")
            continue

        col_tbl, col_fig = st.columns([1, 4])
        with col_tbl:
            if display_mode == "Counts":
                st.dataframe(df_pat[["pattern", "count"]].set_index("pattern"))
            else:
                st.dataframe(
                    (df_pat[["pattern", "pct"]] * 100).round(2).set_index("pattern")
                )
        with col_fig:
            fig, ax = plt.subplots(figsize=(8, 2))
            values = (
                df_pat["count"] if display_mode == "Counts" else df_pat["pct"] * 100
            )
            ax.bar(df_pat["pattern"], values, color="#1f77b4")
            ax.set_xticklabels(df_pat["pattern"], rotation=90)
            ylabel = "Count" if display_mode == "Counts" else "% of rows"
            ax.set_ylabel(ylabel)
            ax.set_title(
                f"{model_name} – {task_choice.capitalize()} ({'g-n-o-f' if include_ground_truth else 'n-o-f'})"
            )
            st.pyplot(fig)

with tab_recall:
    st.subheader("Recall Analysis (classic, cross-prompt & union)")

    metric_options = [
        # classic
        "Hate Recall",
        "Toxicity Recall",
        "Overall Recall",
        # cross-prompt
        "Toxic-on-Hate Recall",
        "Hate-on-Toxic Recall",
        # cross-prompt on union
        "Toxic-on-Either Recall",
        "Hate-on-Either Recall",
        # union of prompts
        "Overall-on-Toxic Recall",
        "Overall-on-Hate Recall",
        "Overall-on-Either Recall",
    ]
    selected_metric = st.sidebar.selectbox("Metric to plot", metric_options)

    # -- SUMMARY TABLE (no bins) --
    summary_rows = []
    for model in MODELS:
        for context in CONTEXTS:
            toxicity_column, hate_column = (
                f"toxicity_{model}_{context}_verdict",
                f"hate_{model}_{context}_verdict",
            )

            total_amount_of_hate_speech = is_hate_speech.sum()
            total_amount_of_toxic_speech = is_toxic_speech.sum()
            total_either = (is_hate_speech | is_toxic_speech).sum()

            hate_speech_tp = (
                df[is_hate_speech]
                .apply(verdict_equals_one, col=hate_column, axis=1)
                .sum()
            )
            toxic_speech_tp = (
                df[is_toxic_speech]
                .apply(verdict_equals_one, col=toxicity_column, axis=1)
                .sum()
            )
            toxicity_prompt_on_hate_speech_tp = (
                df[is_hate_speech]
                .apply(verdict_equals_one, col=toxicity_column, axis=1)
                .sum()
            )
            hate_prompt_on_toxic_speech_tp = (
                df[is_toxic_speech]
                .apply(verdict_equals_one, col=hate_column, axis=1)
                .sum()
            )

            toxicity_on_either_tp = toxic_speech_tp + toxicity_prompt_on_hate_speech_tp
            hate_on_either_tp = hate_speech_tp + hate_prompt_on_toxic_speech_tp

            either_prompt_on_toxic_speech_tp = (
                df[is_toxic_speech]
                .apply(
                    lambda r: verdict_equals_one(r, toxicity_column)
                    or verdict_equals_one(r, hate_column),
                    axis=1,
                )
                .sum()
            )
            either_prompt_on_hate_speech_tp = (
                df[is_hate_speech]
                .apply(
                    lambda r: verdict_equals_one(r, toxicity_column)
                    or verdict_equals_one(r, hate_column),
                    axis=1,
                )
                .sum()
            )
            either_prompt_on_either_tp = (
                df[is_hate_speech | is_toxic_speech]
                .apply(
                    lambda r: verdict_equals_one(r, toxicity_column)
                    or verdict_equals_one(r, hate_column),
                    axis=1,
                )
                .sum()
            )
            overall_tp = hate_speech_tp + toxic_speech_tp
            summary_rows.append(
                {
                    "Model": model.upper(),
                    "Context": context,
                    "Hate Recall": safe_divide(
                        hate_speech_tp, total_amount_of_hate_speech
                    ),
                    "Toxicity Recall": safe_divide(
                        toxic_speech_tp, total_amount_of_toxic_speech
                    ),
                    "Overall Recall": safe_divide(
                        overall_tp,
                        total_amount_of_hate_speech + total_amount_of_toxic_speech,
                    ),
                    "Toxic-on-Hate Recall": safe_divide(
                        toxicity_prompt_on_hate_speech_tp, total_amount_of_hate_speech
                    ),
                    "Hate-on-Toxic Recall": safe_divide(
                        hate_prompt_on_toxic_speech_tp, total_amount_of_toxic_speech
                    ),
                    "Toxic-on-Either Recall": safe_divide(
                        toxicity_on_either_tp, total_either
                    ),
                    "Hate-on-Either Recall": safe_divide(
                        hate_on_either_tp, total_either
                    ),
                    "Overall-on-Toxic Recall": safe_divide(
                        either_prompt_on_toxic_speech_tp, total_amount_of_toxic_speech
                    ),
                    "Overall-on-Hate Recall": safe_divide(
                        either_prompt_on_hate_speech_tp, total_amount_of_hate_speech
                    ),
                    "Overall-on-Either Recall": safe_divide(
                        either_prompt_on_either_tp, total_either
                    ),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    st.subheader("Unified Recall Table (whole dataset)")
    with st.expander("Show / Hide column definitions"):
        st.markdown(
            """
    ### Classic (single-prompt, single-label)
    - **Hate Recall** – % of *hate-labeled* comments flagged by the **hate-speech** prompt  
    - **Toxicity Recall** – % of *toxic-labeled* comments flagged by the **toxicity** prompt  
    - **Overall Recall** – % of *all hate ∪ toxic* comments flagged by their **own** prompt  

    ### Cross-prompt (prompt tested on the *other* label)
    - **Toxic-on-Hate Recall** – Toxicity prompt on hate-labeled comments  
    - **Hate-on-Toxic Recall** – Hate prompt on toxic-labeled comments  

    ### Cross-prompt on the union label
    - **Toxic-on-Either Recall** – Toxicity prompt on *(hate ∪ toxic)* comments  
    - **Hate-on-Either Recall** – Hate prompt on *(hate ∪ toxic)* comments  

    ### Union-of-prompts (either prompt may fire)
    - **Overall-on-Toxic Recall** – % of toxic-labeled comments flagged by **either** prompt  
    - **Overall-on-Hate Recall** – % of hate-labeled comments flagged by **either** prompt  
    - **Overall-on-Either Recall** – % of *(hate ∪ toxic)* comments flagged by **either** prompt
    """
        )

    st.dataframe(summary_df)

    # -- BIN-LEVEL METRICS --
    df_binned = df.copy()
    df_binned["aggregate_bin"] = pd.cut(df_binned["aggregate_avg"], bins=num_bins)
    bin_counts = (
        df_binned.groupby("aggregate_bin", observed=True)
        .size()
        .reset_index(name="n_samples")
        .sort_values("aggregate_bin")
    )

    metric_rows = []
    for bin_interval, bin_df in df_binned.groupby("aggregate_bin"):
        mid = bin_midpoint(bin_interval)
        bin_is_toxic_speech = bin_df["annotation"].str.contains(
            GROUND_TRUTH_TOXIC, na=False
        )
        bin_is_hate_speech = bin_df["annotation"].str.contains(
            GROUND_TRUTH_HATE, na=False
        )

        tox_total_bin, hate_total_bin = (
            bin_is_toxic_speech.sum(),
            bin_is_hate_speech.sum(),
        )

        for model in MODELS:
            for context in CONTEXTS:
                toxicity_column, hate_column = (
                    f"toxicity_{model}_{context}_verdict",
                    f"hate_{model}_{context}_verdict",
                )
                bin_is_either = bin_is_hate_speech | bin_is_toxic_speech
                either_total_bin = bin_is_either.sum()

                either_prompt_on_either_tp_bin = (
                    bin_df[bin_is_either]
                    .apply(
                        lambda r: verdict_equals_one(r, toxicity_column)
                        or verdict_equals_one(r, hate_column),
                        axis=1,
                    )
                    .sum()
                )
                toxic_speech_tp_per_bin = (
                    bin_df[bin_is_toxic_speech]
                    .apply(verdict_equals_one, col=toxicity_column, axis=1)
                    .sum()
                )
                hate_speech_tp_per_bin = (
                    bin_df[bin_is_hate_speech]
                    .apply(verdict_equals_one, col=hate_column, axis=1)
                    .sum()
                )
                toxic_prompt_on_hate_speech_tp_per_bin = (
                    bin_df[bin_is_hate_speech]
                    .apply(verdict_equals_one, col=toxicity_column, axis=1)
                    .sum()
                )
                hate_prompt_on_toxic_speech_tp_per_bin = (
                    bin_df[bin_is_toxic_speech]
                    .apply(verdict_equals_one, col=hate_column, axis=1)
                    .sum()
                )

                tox_on_either_tp_bin = (
                    toxic_speech_tp_per_bin + toxic_prompt_on_hate_speech_tp_per_bin
                )
                hate_on_either_tp_bin = (
                    hate_speech_tp_per_bin + hate_prompt_on_toxic_speech_tp_per_bin
                )

                either_prompt_on_toxic_speech_tp_per_bin = (
                    bin_df[bin_is_toxic_speech]
                    .apply(
                        lambda r: verdict_equals_one(r, toxicity_column)
                        or verdict_equals_one(r, hate_column),
                        axis=1,
                    )
                    .sum()
                )
                either_prompt_on_hate_speech_tp_per_bin = (
                    bin_df[bin_is_hate_speech]
                    .apply(
                        lambda r: verdict_equals_one(r, toxicity_column)
                        or verdict_equals_one(r, hate_column),
                        axis=1,
                    )
                    .sum()
                )
                overall_tp_bin = hate_speech_tp_per_bin + toxic_speech_tp_per_bin

                metric_rows.append(
                    {
                        "Bin Midpoint": mid,
                        "Model": model.upper(),
                        "Context": context,
                        "Hate Recall": safe_divide(
                            hate_speech_tp_per_bin, hate_total_bin
                        ),
                        "Toxicity Recall": safe_divide(
                            toxic_speech_tp_per_bin, tox_total_bin
                        ),
                        "Overall Recall": safe_divide(
                            overall_tp_bin, hate_total_bin + tox_total_bin
                        ),
                        "Toxic-on-Hate Recall": safe_divide(
                            toxic_prompt_on_hate_speech_tp_per_bin, hate_total_bin
                        ),
                        "Hate-on-Toxic Recall": safe_divide(
                            hate_prompt_on_toxic_speech_tp_per_bin, tox_total_bin
                        ),
                        "Toxic-on-Either Recall": safe_divide(
                            tox_on_either_tp_bin, either_total_bin
                        ),
                        "Hate-on-Either Recall": safe_divide(
                            hate_on_either_tp_bin, either_total_bin
                        ),
                        "Overall-on-Toxic Recall": safe_divide(
                            either_prompt_on_toxic_speech_tp_per_bin, tox_total_bin
                        ),
                        "Overall-on-Hate Recall": safe_divide(
                            either_prompt_on_hate_speech_tp_per_bin, hate_total_bin
                        ),
                        "Overall-on-Either Recall": safe_divide(
                            either_prompt_on_either_tp_bin, either_total_bin
                        ),
                    }
                )
    metrics_df = pd.DataFrame(metric_rows)

    # -- BAR CHARTS --
    st.subheader(f"Bar Charts — {selected_metric}")
    columns = st.columns(3)
    for idx, model in enumerate(MODELS):
        container = columns[idx]
        df_m = metrics_df[metrics_df["Model"] == model.upper()].copy()
        pivot = df_m.pivot(
            index="Bin Midpoint", columns="Context", values=selected_metric
        ).sort_index()
        if pivot.empty:
            container.info("Metric unavailable for this model.")
            continue
        x = np.arange(len(pivot.index))
        w = 0.25
        fig, ax = plt.subplots()
        for i, context in enumerate(pivot.columns):
            ax.bar(
                x + i * w,
                pivot[context],
                w,
                color=BAR_COLORS.get(context, "#666"),
                label=context.replace("_", " "),
            )
        ax.set_xticks(x + w * (len(pivot.columns) - 1) / 2)
        ax.set_xticklabels([round(b, 2) for b in pivot.index], rotation=45)
        ax.set_xlabel("Aggregate Avg (Bin Midpoint)")
        ax.set_ylabel("Recall")
        ax.set_ylim(0, 1)
        ax.legend(loc="center right")
        model_name = MODEL_NAMES.get(model, "unknown")
        ax.set_title(model_name)
        container.pyplot(fig)

        # download
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode()
        href = f'<a href="data:image/png;base64,{b64}" download="{model}_{selected_metric.replace(" ", "_")}_{num_bins}_bars.png">Download this plot</a>'
        container.markdown(href, unsafe_allow_html=True)

    # -- DELTA LINE CHARTS --
    columns = st.columns(3)
    for idx, model in enumerate(MODELS):
        container = columns[idx]
        df_m = metrics_df[metrics_df["Model"] == model.upper()].copy()
        pivot = df_m.pivot(
            index="Bin Midpoint", columns="Context", values=selected_metric
        )

        if pivot.empty or "no_context" not in pivot.columns:
            container.info("Metric unavailable for delta plot.")
            continue

        fig, ax = plt.subplots()

        if "with_original" in pivot.columns:
            label_with_original_vs_no_context = "with_original - no_context"
            ax.plot(
                pivot.index,
                pivot["with_original"] - pivot["no_context"],
                marker="o",
                color=LINE_COLORS[label_with_original_vs_no_context],
                label=label_with_original_vs_no_context,
            )

        if "with_full_context" in pivot.columns:
            label_with_full_context_vs_no_context = "with_full_context - no_context"
            ax.plot(
                pivot.index,
                pivot["with_full_context"] - pivot["no_context"],
                marker="o",
                color=LINE_COLORS[label_with_full_context_vs_no_context],
                label=label_with_full_context_vs_no_context,
            )

        ax.set_xticks(pivot.index)
        ax.set_xticklabels([round(b, 2) for b in pivot.index], rotation=45)
        ax.axhline(0, color="gray", linestyle="--")
        ax.set_xlabel("Aggregate Avg (Bin Midpoint)")
        ax.set_ylabel("Recall Delta")
        model_name = MODEL_NAMES.get(model, "unknown")
        ax.set_title(f"{model_name} - {selected_metric} (Δ)")
        ax.legend(loc="center right")
        container.pyplot(fig)

        container.markdown(f"##### Raw Recall Values ({model_name})")
        container.dataframe(pivot.style.format("{:.3f}"), height=250, width="stretch")

        delta_df = pd.DataFrame(index=pivot.index)
        if "with_original" in pivot.columns:
            delta_df["with_original - no_context"] = (
                pivot["with_original"] - pivot["no_context"]
            )
        if "with_full_context" in pivot.columns:
            delta_df["with_full_context - no_context"] = (
                pivot["with_full_context"] - pivot["no_context"]
            )

        container.markdown(f"##### Raw Recall Delta Values ({model_name})")
        container.dataframe(
            delta_df.style.format("{:+.3f}"), height=250, width="stretch"
        )

    # Download all three models’ plots as one image (bar row + delta row)
    combined_buffer = io.BytesIO()
    fig_all, axes = plt.subplots(2, 3, figsize=(18, 8))
    for i, model in enumerate(MODELS):
        df_m = metrics_df[metrics_df["Model"] == model.upper()]
        pivot = df_m.pivot(
            index="Bin Midpoint", columns="Context", values=selected_metric
        ).sort_index()
        x = np.arange(len(pivot.index))
        w = 0.25
        for j, context in enumerate(pivot.columns):
            axes[0, i].bar(
                x + j * w,
                pivot[context],
                w,
                color=BAR_COLORS.get(context, "#666"),
                label=context.replace("_", " "),
            )
        axes[0, i].set_xticks(x + w * (len(pivot.columns) - 1) / 2)
        axes[0, i].set_title(f"{MODEL_NAMES.get(model)} - {selected_metric}")
        axes[0, i].set_xticklabels([round(b, 2) for b in pivot.index], rotation=45)
        axes[0, i].set_xlabel("Aggregate Avg (Bin Midpoint)")
        axes[0, i].set_ylabel("Recall")
        axes[0, i].set_ylim(0, 1)
        axes[0, i].legend(loc="center right")

        pivot2 = df_m.pivot(
            index="Bin Midpoint", columns="Context", values=selected_metric
        )
        if "with_original" in pivot2:
            label_with_original_vs_no_context = "with_original - no_context"
            axes[1, i].plot(
                pivot2.index,
                pivot2["with_original"] - pivot2["no_context"],
                marker="o",
                color=LINE_COLORS[label_with_original_vs_no_context],
                label=label_with_original_vs_no_context,
            )
        if "with_full_context" in pivot2:
            label_with_full_context_vs_no_context = "with_full_context - no_context"
            axes[1, i].plot(
                pivot2.index,
                pivot2["with_full_context"] - pivot2["no_context"],
                marker="o",
                color=LINE_COLORS[label_with_full_context_vs_no_context],
                label=label_with_full_context_vs_no_context,
            )
        axes[1, i].set_xticks(pivot.index)
        axes[1, i].set_xticklabels([round(b, 2) for b in pivot.index], rotation=45)
        axes[1, i].axhline(0, color="gray", linestyle="--")
        axes[1, i].set_xlabel("Aggregate Avg (Bin Midpoint)")
        axes[1, i].set_ylabel("Recall Delta")
        axes[1, i].set_title(f"{MODEL_NAMES.get(model)} - {selected_metric} (Δ)")
        axes[1, i].legend(loc="center right")

    plt.tight_layout()
    fig_all.savefig(combined_buffer, format="png", bbox_inches="tight", dpi=300)
    combined_buffer.seek(0)
    b64_all = base64.b64encode(combined_buffer.read()).decode()
    href_all = f'<a href="data:image/png;base64,{b64_all}" download="all_models_{selected_metric.replace(" ", "_")}_{num_bins}_combined_plots.png">Download all plots</a>'
    st.markdown(href_all, unsafe_allow_html=True)
