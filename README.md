# Beyond RFM: Product-Aware Household Segmentation in Grocery Retail

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dothanhchuong123-blip/beyond-rfm-grocery-segmentation/blob/main/grocery_clustering.ipynb)

Mid-term project, **Topic 10: Clustering on the Grocery Store Dataset (Transaction & Product)**. It uses dunnhumby's *The Complete Journey*.

**Author:** Đỗ Thành Chương (11251670) · **Instructor:** Trần Đức Minh

📄 **Report (PDF):** [`output/pdf/Beyond_RFM_Report.pdf`](output/pdf/Beyond_RFM_Report.pdf)
📓 **Notebook:** [`grocery_clustering.ipynb`](grocery_clustering.ipynb)

---

## Research question

Does adding shopping behavior and product (department) preferences give a better household segmentation than a classic **RFM** (Recency, Frequency, Monetary) baseline?

Clustering has no ground-truth labels, so each candidate segmentation is judged on four things:

1. **Geometric fit**: silhouette score and the elbow method
2. **Stability**: Adjusted Rand Index (ARI) across re-initializations and 80% subsamples
3. **Interpretability**: cluster sizes and spending/category profiles
4. **Later-period validation**: whether clusters fitted on calibration data (days ≤ 500) still separate spending in the holdout window (days 501–711)

## Pipeline

| Step | What happens |
|---|---|
| Data inspection | Schema and UTF-8 encoding audit of all 8 CSVs. Primary/foreign keys checked (`PRODUCT_ID`, `household_key`, `BASKET_ID`) |
| Loading & integration | Validated many-to-one join of `transaction_data.csv` → `product.csv`. Row count must match the transaction count |
| Cleaning | Duplicate and missing-value audit. 18,917 lines with nonpositive quantity or sales removed (2,576,815 kept). Blank departments set to `UNKNOWN` |
| EDA | Distributions, correlations and department mix, used to choose features |
| Features | **RFM** (3) → **Behavior** (8: adds basket size, product variety, discount usage) → **Product** (17: adds department shares). log1p + standardization, fitted on the calibration period only |
| Models | **K-means** (k = 2…8, baseline), **Hierarchical / Ward**, **DBSCAN** |
| Evaluation | Silhouette, elbow, ARI stability, PCA projection, holdout η² with bootstrap intervals |

## Key results

- All three feature spaces pick **k = 2**, two broad activity groups (higher-spend vs. lower-spend households).
- **RFM is the best default.** It has the highest stability (mean subsample ARI 0.980, vs. 0.970 for Behavior and 0.964 for Product). It also explains later spending at least as well (η² 0.204 vs. 0.202). At k = 3 its lead grows (η² 0.313 vs. 0.170).
- K-means is much more stable than Ward (ARI ≈ 0.96–0.98 vs. 0.31–0.54). DBSCAN gives no useful multi-cluster partition.
- Product-aware features move **287 of 2,498 households (11.5%)** to the other group. However, the gain in later category separation is small and not significant: +0.087 pp, 95% CI [−0.110, +0.315].
- **Takeaway:** a simple RFM segmentation is the most reliable. Product features are best used to *describe* segments, not to *define* them.

Later-period differences are descriptive and do not prove that targeting by these segments would lift sales.

## Repository structure

```
├── grocery_clustering.ipynb   # Main notebook: runs end to end (Jupyter or Colab)
├── src/
│   ├── analysis.py            # Same analysis as a standalone script
│   └── schema_audit.py        # Optional schema/encoding inventory of all 8 CSVs
├── figures/                   # Generated charts (EDA, model selection, stability, PCA, …)
├── results/                   # Generated tables (metrics, profiles, stability, provenance)
├── report/                    # LaTeX source of the report (+ references.bib, logos)
├── output/pdf/                # Compiled report PDF
├── docs/                      # Rubric checklist
├── requirements.txt           # Dependencies
└── requirements-lock.txt      # Exact validated package versions
```

## How to run

### Option A: Google Colab

Click the **Open in Colab** badge above, then choose **Runtime → Run all**. The notebook installs any missing packages and downloads the public dataset (~124 MB) automatically.

### Option B: Local Jupyter (Python 3.12 recommended)

```bash
git clone https://github.com/dothanhchuong123-blip/beyond-rfm-grocery-segmentation.git
cd beyond-rfm-grocery-segmentation
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt jupyterlab
jupyter lab grocery_clustering.ipynb
```

Then choose **Kernel → Restart Kernel and Run All**. Run from the repository root. The notebook regenerates `figures/` and `results/`.

To run the script version instead:

```bash
python src/analysis.py
```

### Data

The raw data is **not committed** because of its size and the provider's terms. The notebook downloads it from [Kaggle: dunnhumby – The Complete Journey](https://www.kaggle.com/datasets/frtgnn/dunnhumby-the-complete-journey) if it is missing. If the download fails, get the dataset manually and put `transaction_data.csv` and `product.csv` in `data/raw/`. Joining about 2.6 M transaction rows needs a few GB of RAM.

### Rebuild the report (optional)

```bash
tectonic report/main.tex --outdir output/pdf
```

This needs [Tectonic](https://tectonic-typesetting.github.io/) or any XeLaTeX distribution.

## Data source

dunnhumby, *The Complete Journey*: [dunnhumby Source Files](https://www.dunnhumby.com/source-files/). The dataset was used for academic purposes only.
