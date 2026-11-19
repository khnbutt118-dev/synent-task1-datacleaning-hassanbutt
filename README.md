<p align="center"><img src="https://raw.githubusercontent.com/khnbutt118-dev/synent-task1-datacleaning-hassanbutt/main/assets/project-banner.svg" width="100%" alt="DATA CLEANING — Data quality · preprocessing · Titanic dataset"></p>
<p align="center"><img src="https://img.shields.io/badge/SYNENT-Task%2001-087EAD?style=for-the-badge"><img src="https://img.shields.io/badge/Focus-Data%20Quality-155E9A?style=for-the-badge"><img src="https://img.shields.io/badge/Level-Basic-0B719E?style=for-the-badge"></p>
<p align="center"><a href="https://github.com/khnbutt118-dev">← Profile</a> · <a href="https://hassanbuttportfolio.vercel.app/">Portfolio</a> · <a href="https://www.linkedin.com/in/hassan-butt-ab0036334/">LinkedIn</a></p>

# 🧹 Titanic · Data Cleaning & Preprocessing
**Synent Technologies · Data Science Internship · Task 1**

<table><tr><td align="center"><h2>891</h2><sub>Rows retained</sub></td><td align="center"><h2>866 → 0</h2><sub>Missing cells</sub></td><td align="center"><h2>0</h2><sub>Duplicate rows removed</sub></td></tr></table>

## The transformation

| Data quality check | Before | After |
|:--|--:|--:|
| Rows | 891 | 891 |
| Missing cells | 866 | **0** |
| Duplicate rows removed | — | 0 |

## 🛠️ What changed

- Standardised column names to snake_case.
- Removed duplicate rows and duplicate passenger IDs (none were present).
- Imputed Age using median by passenger class and sex; Embarked using mode.
- Converted Cabin into `has_cabin` and `deck` features rather than imputing a column with 77% missing values.
- Corrected data types and generated a before/after quality report.

## 📈 Visual results

<p align="center"><img src="outputs/missing_values_and_age.png" alt="Missing values and age analysis" width="80%"></p>
<p align="center"><img src="outputs/survival_after_cleaning.png" alt="Survival analysis after cleaning" width="80%"></p>

**Exploratory observation:** survival in this dataset was 74% for women and 19% for men. This is a descriptive result for this dataset, not a causal conclusion.

## ▶️ Run

```bash
pip install -r requirements.txt
python main.py
```

<details><summary><b>Dataset & files</b></summary>

- Dataset: [Titanic passenger dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset)
- 891 rows × 12 columns
- Outputs: `titanic_clean.csv`, `data_quality_report.json`
- Main files: `main.py`, `requirements.txt`, `outputs/`
</details>

---
<p align="center"><sub>Clean data is the foundation of reliable analysis.</sub></p>