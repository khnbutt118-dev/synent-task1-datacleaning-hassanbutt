"""Synent Data Science Internship - Task 1: Data Cleaning & Preprocessing (Titanic)
Author: Hassan Butt
Run: python main.py
"""

import sys, os, re, json, shutil, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")


DATASETS = {'titanic': {'files': ['titanic.csv', 'Titanic-Dataset.csv', 'train.csv'],
             'kaggle': 'yasserh/titanic-dataset',
             'name': 'Titanic passenger dataset',
             'page': 'https://www.kaggle.com/datasets/yasserh/titanic-dataset',
             'url': 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'}}



def _outdir(repo_dir):
    p = Path(repo_dir) / "outputs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_fig(fig, repo_dir, name):
    fig.savefig(_outdir(repo_dir) / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return name


def save_json(obj, repo_dir, name):
    (_outdir(repo_dir) / name).write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def _read_csv(path, spec):
    kw = {"encoding": spec.get("encoding", "utf-8")}
    if spec.get("no_header_file") and Path(path).name == spec["no_header_file"]:
        kw["header"] = None
        kw["names"] = spec["names"]
    try:
        return pd.read_csv(path, **kw)
    except UnicodeDecodeError:
        kw["encoding"] = "latin-1"
        return pd.read_csv(path, **kw)


def _builtin(key):
    if key == "iris":
        from sklearn.datasets import load_iris
        d = load_iris(as_frame=True)
        df = d.frame.drop(columns=["target"])
        df["species"] = [d.target_names[i] for i in d.target]
        return df
    raise KeyError(key)


def get_data(key, repo_dir):
    """Load a dataset: local data/ folder -> built-in -> kagglehub -> public URL."""
    spec = DATASETS[key]
    data_dir = Path(repo_dir) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in spec["files"]:
        p = data_dir / name
        if p.exists():
            print(f"  [data] using local file {p}")
            return _read_csv(p, spec)
    if spec.get("builtin"):
        print("  [data] using built-in copy")
        return _builtin(key)
    if spec.get("kaggle"):
        try:
            import kagglehub
            folder = Path(kagglehub.dataset_download(spec["kaggle"]))
            cands = list(folder.rglob("*.csv"))
            pick = None
            for want in spec["files"]:
                for c in cands:
                    if c.name.lower() == want.lower():
                        pick = c
                        break
                if pick:
                    break
            if pick is None and cands:
                pick = cands[0]
            if pick is not None:
                dest = data_dir / pick.name
                shutil.copy(pick, dest)
                print(f"  [data] downloaded from Kaggle: {dest}")
                return _read_csv(dest, spec)
        except Exception as e:
            print(f"  [data] kagglehub download not available ({type(e).__name__}: {e})")
    if spec.get("url"):
        try:
            import urllib.request
            dest = data_dir / (spec.get("no_header_file") or spec["files"][0])
            urllib.request.urlretrieve(spec["url"], dest)
            print(f"  [data] downloaded from public URL: {dest}")
            return _read_csv(dest, spec)
        except Exception as e:
            print(f"  [data] public URL download failed ({type(e).__name__}: {e})")
    raise FileNotFoundError(
        f"Dataset '{spec['name']}' not found.\n"
        f"  -> Download it from {spec['page']}\n"
        f"  -> Save the CSV as one of {spec['files']} inside: {data_dir.resolve()}\n"
        f"  -> Then run the script again.")


def _to_datetime(series):
    try:
        return pd.to_datetime(series, errors="coerce", format="mixed")
    except (ValueError, TypeError):
        return pd.to_datetime(series, errors="coerce")


def _snake(col):
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(col).strip()).strip("_").lower()


def task1_run(repo_dir):
    raw = get_data("titanic", repo_dir)
    df = raw.copy()
    m = {"dataset_shape": [int(raw.shape[0]), int(raw.shape[1])]}
    miss_before = raw.isna().sum()
    m["missing_before"] = {k: int(v) for k, v in miss_before[miss_before > 0].items()}
    m["duplicates_before"] = int(raw.duplicated().sum())

    # 1) Rename columns -> clean snake_case names
    df.columns = [_snake(c) for c in df.columns]
    df = df.rename(columns={"pclass": "passenger_class", "sibsp": "siblings_spouses",
                            "parch": "parents_children", "embarked": "port_embarked",
                            "passengerid": "passenger_id"})

    # 2) Remove duplicates
    n0 = len(df)
    df = df.drop_duplicates()
    if "passenger_id" in df.columns:
        df = df.drop_duplicates(subset="passenger_id")
    m["duplicates_removed"] = int(n0 - len(df))

    # 3) Handle missing values
    pct_cabin = float(raw["Cabin"].isna().mean() * 100) if "Cabin" in raw.columns else 0.0
    if "cabin" in df.columns:
        df["has_cabin"] = df["cabin"].notna().astype(int)
        df["deck"] = df["cabin"].astype("string").str[0].fillna("Unknown")
        df = df.drop(columns=["cabin"])
    if "name" in df.columns:
        df["title"] = df["name"].astype("string").str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
        rare = df["title"].value_counts()
        rare = rare[rare < 10].index
        df["title"] = df["title"].where(~df["title"].isin(rare), "Rare")
    age_missing_pct = float(df["age"].isna().mean() * 100)
    df["age"] = df["age"].fillna(df.groupby(["passenger_class", "sex"])["age"].transform("median"))
    df["age"] = df["age"].fillna(df["age"].median())
    if "port_embarked" in df.columns:
        df["port_embarked"] = df["port_embarked"].fillna(df["port_embarked"].mode().iloc[0])
    if df["fare"].isna().any():
        df["fare"] = df["fare"].fillna(df.groupby("passenger_class")["fare"].transform("median"))
    m["missing_after"] = int(df.isna().sum().sum())

    # 4) Convert data types
    df["passenger_id"] = df["passenger_id"].astype(int)
    df["survived"] = df["survived"].astype(int)
    df["passenger_class"] = pd.Categorical(df["passenger_class"], categories=[1, 2, 3], ordered=True)
    for c in ["sex", "port_embarked", "deck", "title"]:
        if c in df.columns:
            df[c] = df[c].astype("category")
    df["age"] = df["age"].astype(float).round(1)
    df["fare"] = df["fare"].astype(float).round(2)
    for c in ["siblings_spouses", "parents_children"]:
        df[c] = df[c].astype(int)
    m["dtypes"] = {c: str(t) for c, t in df.dtypes.items()}
    m["rows_clean"] = int(len(df))

    out = _outdir(repo_dir)
    df.to_csv(out / "titanic_clean.csv", index=False)

    # Charts: missing values before cleaning + Age before/after imputation
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    mp = (raw.isna().mean() * 100).sort_values(ascending=False)
    mp = mp[mp > 0]
    sns.barplot(x=mp.values, y=mp.index, ax=ax[0], color="#c0392b")
    ax[0].set_title("Missing values BEFORE cleaning (%)")
    ax[0].set_xlabel("% missing")
    ax[0].set_ylabel("")
    sns.histplot(raw["Age"].dropna(), bins=30, color="#7f8c8d", label="original (non-missing)", ax=ax[1], stat="density")
    sns.histplot(df["age"], bins=30, color="#2980b9", alpha=0.45, label="after imputation", ax=ax[1], stat="density")
    ax[1].set_title("Age distribution before vs after imputation")
    ax[1].legend()
    save_fig(fig, repo_dir, "missing_values_and_age.png")

    # Quick EDA on the cleaned data (for insights)
    surv_sex = df.groupby("sex", observed=True)["survived"].mean() * 100
    surv_cls = df.groupby("passenger_class", observed=True)["survived"].mean() * 100
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    sns.barplot(x=surv_sex.index.astype(str), y=surv_sex.values, ax=ax[0], palette="Blues_d")
    ax[0].set_title("Survival rate by sex (%)")
    sns.barplot(x=surv_cls.index.astype(str), y=surv_cls.values, ax=ax[1], palette="Greens_d")
    ax[1].set_title("Survival rate by passenger class (%)")
    save_fig(fig, repo_dir, "survival_after_cleaning.png")

    report = {k: v for k, v in m.items() if k != "dtypes"}
    report["dtypes"] = m["dtypes"]
    save_json(report, repo_dir, "data_quality_report.json")

    insights = [
        f"Cabin was {pct_cabin:.0f}% missing, so it was converted into 'has_cabin' and 'deck' features instead of being imputed.",
        f"Age was {age_missing_pct:.0f}% missing and was filled with the median age of passengers in the same class and sex group.",
        f"{m['duplicates_removed']} duplicate rows were removed and all {sum(m['missing_before'].values())} missing cells were handled (0 left).",
        f"After cleaning, survival was {surv_sex.get('female', float('nan')):.0f}% for women vs {surv_sex.get('male', float('nan')):.0f}% for men.",
    ]
    return {"headline": {"Rows (raw -> clean)": f"{m['dataset_shape'][0]} -> {m['rows_clean']}",
                         "Missing cells (before -> after)": f"{sum(m['missing_before'].values())} -> {m['missing_after']}",
                         "Duplicates removed": m["duplicates_removed"]},
            "insights": insights, "dataset_shape": m["dataset_shape"],
            "files": ["titanic_clean.csv", "data_quality_report.json", "missing_values_and_age.png",
                      "survival_after_cleaning.png"]}



if __name__ == "__main__":
    res = task1_run(Path(__file__).parent)
    print(json.dumps(res, indent=2, default=str))
