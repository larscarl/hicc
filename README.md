[![License: MIT](https://img.shields.io/badge/Code%20License-MIT-yellow.svg)](./LICENSE)
[![Data: CC BY-NC 4.0](https://img.shields.io/badge/Data%20License-CC%20BY--NC%204.0-lightgrey.svg)](./LICENSE-DATA)

# HICC: A Dataset for German Hate Speech in Conversational Context

**Post IDs + annotations for context-aware hate/toxic speech research**

> **Content warning:** This dataset contains references to hateful, toxic, and otherwise harmful language.

## What is HICC?
HICC pairs German social-media comments with their **conversational context** (root post + all direct replies to the root published before the target) to help study hateful/toxic speech that is only detectable **with context**. The data was collected by trained digital streetworkers and manually annotated; details of the collection protocol, definitions, and evaluation appear in the paper.

**At a glance:**
- **21,338** posts total; **1,812** annotated "intervention" targets (940 toxic, 734 hate, 108 "intervention not recommended", 30 "no intervention").
- Context windows average **~11.8** posts per target.
- LLM classification with **full prior context** improves recall for hard cases (up to **+19 pp**).
- To comply with [X policy](https://developer.x.com/en/support/x-api/policy), we distribute **Post IDs/URLs + annotations**, not text. Rehydration is required.

## Files
- `hicc_dataset.csv` — core table (Post IDs/URLs, annotations, scores, model verdicts).
- `evaluate_recall.py` – a Streamlit app that reads `hicc_dataset.csv` and provides analyses of recall and label flips (run with `streamlit run evaluate_recall.py`).
- `rehydrate_tweets.py` — utility to reconstruct tweet text into a working copy (**see Quickstart**).
- `paper/HICC.pdf` — camera-ready PDF of the paper accepted at KONVENS 2025 for an oral presentation.

**Policy note (X / formerly Twitter):** Redistribution must be limited to IDs and derived metadata. Content must be rehydrated via the official API subject to access rights. Please follow the current X policy linked above.

## Schema (selected columns)
Each row is a post in a reconstructed thread.

- `id` — internal numeric identifier.
- `annotator` — anonymized annotator ID for targets.
- `type` — `"CONTEXT"` (root) or `"COMMENT"` (direct replies to the root).
- `post_time_exact` — ISO 8601 timestamp (`YYYY-MM-DDTHH:mm:ss.sssZ`) if available for comments.
- `x_url` — URL to the post on X (used to extract the Post ID).
- `is_annotated_text` — `1` for the chosen target in a thread, else `0`.
- `annotation` — labels for targets, e.g. `['INTERVENTION_TOXIC']`, `['INTERVENTION_HATE']`, `['INTERVENTION_NOT_RECOMMENDED']`, `['NO_INTERVENTION']` (see paper for exact definitions).

**Automatic moderation scores (no context, target text only):**  
- `perspective_toxicity`, `perspective_severe_toxicity` (https://www.perspectiveapi.com/)
- `openai_hate_score`, `openai_hate` (https://platform.openai.com/docs/guides/moderation)
- `hyssop_toxicity`, `hyssop_hate_speech` (SwissBERT-based internal model)
- `aggregate_sum` — sum of the automatic scores.
- `aggregate_avg` — mean of the automatic scores.

**LLM verdicts (binary) + rationales (free text), per task and context:**  
- Tasks: `hate_…`, `toxicity_…`
- Providers: `openai` (GPT-4o-mini), `xai` (Grok-3-mini), `claude` (Claude 3 Haiku)
- Contexts:
  - `_no_context_…` (target text only)
  - `_with_original_…` (root + target)
  - `_with_full_context_…` (root + prior direct replies + target)
- Example fields:
  - `hate_openai_no_context_verdict` ∈ {0,1}
  - `toxicity_claude_with_full_context_verdict` ∈ {0,1}
  - `hate_xai_with_original_reasoning` (string)

See paper for prompting and model details.

## Quickstart: reconstruction of tweet texts
`rehydrate_tweets.py` fills a `msg` column with tweet text in a working copy of the CSV. Distribution of text is not included in `hicc_dataset.csv` due to [X policy](https://developer.x.com/en/support/x-api/policy).

### 1) Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Set credentials

Create a `.env` file (or use the template `.env.example`):
```
TWITTER_BEARER_TOKEN = "YOUR_TOKEN_HERE"
```

### 3) Run

```bash
python rehydrate_tweets.py hicc_dataset.csv hicc_dataset_rehydrated.csv
```

The script extracts IDs from `x_url`, batches calls (≤100 IDs), waits on rate limits, resumes from an existing output, and writes texts to `msg`. (See the script for details.)

## How to use

* **Context evaluation:** For each target, use the provided context window as defined above when probing models. Context helps most for low-score ("hard") targets.
* **Labels:** We provide positives (targets needing intervention). Non-targets in the same thread are unlabeled and not guaranteed negatives; avoid naive accuracy/precision estimates without additional annotation.

## Ethical use

For research only. Handle and present examples responsibly; include content warnings in publications and demos. See paper for definitions and annotation guidelines.

## Cite

If you use HICC, please cite the paper:
> **HICC: A Dataset for German Hate Speech in Conversational Context.**
Lars Schmid, Pius von Däniken, Patrick Giedemann, Don Tuggener, Judith Bühler, Maria Kamenowski, Katja Girschik, Laurent Sedano, Dirk Baier, Mark Cieliebak. KONVENS 2025.

## License

**Code:** MIT. See `LICENSE`.

**Dataset & annotations:** CC BY-NC 4.0. See `LICENSE-DATA`. You may share/adapt for non-commercial purposes with attribution and indication of changes.

**Paper** (`paper/HICC.pdf`): licensed/handled by KONVENS 2025; **not** covered by the above.

**Third-party content (rehydrated tweets)**: not redistributed here. Any text you fetch from X/Twitter remains subject to X’s terms and the original authors’ rights; it is **not** covered by our CC license.

### How to attribute the dataset
> Schmid, L., von Däniken, P., Giedemann, P., Tuggener, D., Bühler, J., Kamenowski, M., Girschik, K., Sedano, L., Baier, D., Cieliebak, M. (2025). *HICC: A Dataset for German Hate Speech in Conversational Context.* KONVENS 2025.
> "Contains data © 2025 HICC authors, licensed **CC BY-NC 4.0**."

## Contact

* Lars Schmid: shmr@zhaw.ch
* Don Tuggener: tuge@zhaw.ch