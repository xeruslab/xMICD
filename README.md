# xMICD: Explainable Representation of Multiple ICD Codes

xMICD is a lightweight and interpretable method for representing **multiple ICD codes** as a single low‑dimensional vector. The method is designed to balance **interpretability**, **transparency**, and **predictive usefulness**, making it suitable for clinical machine‑learning pipelines, health‑data research, and reproducible studies.

Unlike hard group‑based encodings or black‑box deep learning models, xMICD uses a **soft‑assignment mechanism**: each ICD code is represented by its similarity to a set of clinically meaningful **anchors** (group‑level prototype vectors). These similarities are normalized and aggregated to produce patient‑level representations that remain easy to inspect and explain.

---

## Key Features

- Supports **single ICD code** and **multiple ICD code** representations
- Anchors can be constructed from arbitrary groupings (e.g. ICD‑10 blocks, CCS, ECI)
- Two anchor construction strategies: **similarity‑based** and **mean‑based**
- Multiple aggregation strategies for multi‑code inputs
- Optional fallback to **parent ICD codes** when embeddings are missing
- Fully deterministic and reproducible once embeddings and anchors are fixed

---

## Requirements

- Python ≥ 3.10
- numpy
- pandas
- scikit‑learn
- clean‑text

```bash
pip install numpy pandas scikit-learn clean-text
```

---

## Embedding Prerequisite

xMICD operates on **pre‑computed ICD embeddings**. The embedding framework itself is not bundled with this repository.

### Using ICD2Vec (recommended example)

ICD2Vec is one possible embedding framework that can be used with xMICD.
It must be obtained separately:

https://github.com/YeongChanLee/ICD2Vec

Within the ICD2Vec repository, locate one of the following files:

- `icd_code_vec_GatorTron-OG_finetuning.pkl`
- `icd_code_vec_GatorTron-OG_finetuning_20230324.pkl`

Convert the `.pkl` file to `.csv` using the provided utility:

```text
ICD2Vec/
├── code/
│   └── 3.pkl2csv.py
└── model/
    └── GatorTron-OG_icd2vec_finetuning/
```

### Required embedding CSV format

If another embedding framework is used, the CSV file **must** follow this structure:

- One column containing ICD codes
- Multiple numeric embedding columns with a shared prefix (e.g. `Embedding_`)

Example:

| ICD_code | Embedding_0 | Embedding_1 | ... |
|---------:|------------:|------------:|-----|
| A00      | 0.021       | -0.113      | ... |

---

## Grouping File

A grouping file defines how ICD codes are mapped to clinical groups.

Required format:

- Column 1: ICD code
- Column 2: Group name (string; no special characters recommended)

Example:

| ICD_code | Group |
|---------:|-------|
| A00      | GI    |
| A01      | GI    |

Any grouping scheme may be used (ICD‑10 blocks, CCS, ECI, or custom designs).

---

## Installation / Import

Clone the repository and import the main class:

```python
from xMICD import xMICD
```

---

## Workflow Overview

There are **two supported workflows**:

1. Build anchors from embeddings + groupings
2. Build xMICD directly from pre‑computed anchors

---

## Workflow 1: Build from Groupings

### Step 1: Initialize

```python
import pandas as pd
from xMICD import xMICD

embedding_df = pd.read_csv("icd_code_vec.csv")
grouping_df  = pd.read_csv("grouping.csv")

xmicd = xMICD()
```

### Step 2: Build anchors

```python
xmicd.build_from_grouping(
    embedding_df=embedding_df,
    grouping_df=grouping_df,
    embeddings_icd_column="ICD_code",
    embeddings_embedding_column_prefix="Embedding_",
    groupings_icd_column="ICD_code",
    groupings_group_column="Group",
    method="similarity",        # "similarity" or "mean"
    use_parent_embedding=False
)
```

#### Anchor construction methods

- **similarity**: selects the ICD code whose embedding has the highest average cosine similarity to other codes in the same group
- **mean**: uses the arithmetic mean of embeddings within the group

---

## Workflow 2: Build from Pre‑computed Anchors

If anchor vectors have already been computed (e.g. for reuse or reproducibility), xMICD can be initialized directly.

```python
xmicd.build_from_anchor(
    embedding_df=embedding_df,
    anchor_df=anchor_df,
    embeddings_icd_column="ICD_code",
    embeddings_embedding_column_prefix="Embedding_",
    anchor_icd_column="ICD_code",
    anchor_embedding_column_prefix="Embedding_"
)
```

---

## Single ICD Code Representation

```python
valid_icds, vectors = xmicd.get_icd_vector([
    "A00", "A01"
])
```

- `valid_icds`: ICD codes that exist in the embedding database
- `vectors`: xMICD vectors with dimension = number of anchors

### Normalization

Cosine similarities are **min–max normalized per ICD code**, ensuring all xMICD values lie in `[0, 1]`.

---

## Multi‑ICD Code Aggregation

```python
valid_icds, vector = xmicd.get_aggregated_vector(
    ["A00", "A01", "B20"],
    method="max"   # default
)
```

### Aggregation methods

- **max** (default): takes the maximum value per anchor
- **avg**: averages values across ICD codes
- **avg3top**: averages the top‑3 values per anchor

---

## Parent ICD Fallback

If `use_parent_embedding=True` is specified, ICD codes missing embeddings will be replaced by the nearest parent code if available.

Example:

```text
S52.001A → S52.0 → S52
```

This option is available in:

- `build_from_grouping`
- `get_icd_vector`
- `get_aggregated_vector`

---

## Output Interpretation

Each xMICD dimension corresponds to a **clinical group anchor**.
Higher values indicate stronger similarity between the ICD code(s) and that group.

This structure enables:

- Feature‑level interpretability
- Group‑wise contribution analysis
- Compatibility with classical ML models
