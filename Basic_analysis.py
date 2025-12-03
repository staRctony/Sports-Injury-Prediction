# upgraded_analysis_and_visualizations_FIXED_v2.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

# ---------- CONFIG ----------
DATASET_1_PATH = "/Users/sudhir/Desktop/Fall2025/Patient_Data/Assignment/Assignment3/Datasets/collegiate_athlete_injury_dataset.csv"
DATASET_2_PATH = "/Users/sudhir/Desktop/Fall2025/Patient_Data/Assignment/Assignment3/Datasets/sports_injury_detection_dataset.csv"

OUTPUT_DIR_1 = "/Users/sudhir/Desktop/Fall2025/Patient_Data/Assignment/Assignment3/plots/dataset_1"
OUTPUT_DIR_2 = "/Users/sudhir/Desktop/Fall2025/Patient_Data/Assignment/Assignment3/plots/dataset_2"
os.makedirs(OUTPUT_DIR_1, exist_ok=True)
os.makedirs(OUTPUT_DIR_2, exist_ok=True)

CLEANED_PATH_1 = "/Users/sudhir/Desktop/Fall2025/Patient_Data/Assignment/Assignment3/Datasets/collegiate_athlete_injury_dataset_cleaned.csv"
CLEANED_PATH_2 = "/Users/sudhir/Desktop/Fall2025/Patient_Data/Assignment/Assignment3/Datasets/sports_injury_detection_dataset_cleaned.csv"

TARGET_1 = "Injury_Indicator"
TARGET_2 = "Injury_Occurred"

# ---------- SAVE FIG ----------
def save_fig(fig, filename, folder):
    fig.tight_layout()
    fig.savefig(os.path.join(folder, filename), dpi=300)
    plt.close(fig)

# ---------- TEXT SAVE ----------
def print_and_save_text(text, filename, folder):
    fpath = os.path.join(folder, filename)
    with open(fpath, "w") as f:
        f.write(text)
    print(text)

# ---------- IQR OUTLIERS ----------
def iqr_outliers(df, col):
    if df[col].dropna().empty:
        return pd.DataFrame()
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
    return df[(df[col] < lower) | (df[col] > upper)]

# ---------- LOAD DATA ----------
df1 = pd.read_csv(DATASET_1_PATH)
df2 = pd.read_csv(DATASET_2_PATH)

# ---------- REMOVE DATETIME COLUMNS ----------
def drop_datetime(df):
    dt_cols = df.select_dtypes(include=["datetime64", "datetime64[ns]", "datetime"]).columns
    return df.drop(columns=dt_cols)

# Convert and drop datetime
if "Session_Date" in df1.columns:
    df1["Session_Date"] = pd.to_datetime(df1["Session_Date"], errors="coerce")
if "Session_Date" in df2.columns:
    df2["Session_Date"] = pd.to_datetime(df2["Session_Date"], errors="coerce")

df1_no_dt = drop_datetime(df1)
df2_no_dt = drop_datetime(df2)

# ---------- BASIC REPORT ----------
def basic_reports(df, folder, name):
    missing = df.isnull().sum()
    print_and_save_text(f"Missing Values:\n{missing}", f"{name}_missing.txt", folder)

    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    txt = "Top 5 Categories:\n"
    for c in cat_cols:
        vc = df[c].value_counts().head(5)
        txt += f"\n{c}:\n{vc}\n"
    print_and_save_text(txt, f"{name}_top5.txt", folder)

    df.describe().T.to_csv(os.path.join(folder, f"{name}_numeric_summary.csv"))

basic_reports(df1, OUTPUT_DIR_1, "dataset_1")
basic_reports(df2, OUTPUT_DIR_2, "dataset_2")

# ---------- ROUNDING FIX ----------
round_cols = [
    "Skin_Temperature_C",
    "Blood_Oxygen_Level_Percent",
    "Impact_Force_Newtons",
    "Cumulative_Fatigue_Index",
    "Injury_Risk_Score"
]

for c in round_cols:
    if c in df2.columns:
        df2[c] = pd.to_numeric(df2[c], errors="coerce").round(2)

# ---------- OUTLIER REPORT ----------
def outlier_report(df, folder, name):
    txt = "Outlier Report (IQR):\n"
    for col in df.select_dtypes(include=np.number).columns:
        out = iqr_outliers(df, col)
        if len(out) > 0:
            txt += f"{col}: {len(out)} outliers\n"
    print_and_save_text(txt, f"{name}_outliers.txt", folder)

outlier_report(df1_no_dt, OUTPUT_DIR_1, "dataset_1")
outlier_report(df2_no_dt, OUTPUT_DIR_2, "dataset_2")

# ---------- HISTOGRAM + BOXPLOT ----------
def hist_and_box(df, folder, name):
    for col in df.select_dtypes(include=np.number).columns:
        s = df[col].dropna()
        if s.empty: continue

        fig, ax = plt.subplots(figsize=(6,4))
        ax.hist(s, bins=20)
        ax.set_title(f"{col} Distribution")
        save_fig(fig, f"{name}_hist_{col}.png", folder)

        fig, ax = plt.subplots(figsize=(5,4))
        ax.boxplot(s)
        ax.set_title(f"{col} Boxplot")
        save_fig(fig, f"{name}_box_{col}.png", folder)

hist_and_box(df1_no_dt, OUTPUT_DIR_1, "dataset_1")
hist_and_box(df2_no_dt, OUTPUT_DIR_2, "dataset_2")

# ---------- CATEGORY BARS ----------
def category_bars(df, folder, name):
    for col in df.select_dtypes(include=['object', 'category']).columns:
        vc = df[col].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(7,4))
        ax.bar(vc.index.astype(str), vc.values)
        ax.set_title(f"{col} Counts")
        plt.xticks(rotation=45)
        save_fig(fig, f"{name}_bar_{col}.png", folder)

category_bars(df1_no_dt, OUTPUT_DIR_1, "dataset_1")
category_bars(df2_no_dt, OUTPUT_DIR_2, "dataset_2")

# ---------- CORRELATION ----------
def corr_heatmap(df, folder, name):
    num_df = df.select_dtypes(include=np.number)
    if num_df.shape[1] < 2:
        return
    corr = num_df.corr()

    fig, ax = plt.subplots(figsize=(8,7))
    cax = ax.imshow(corr, cmap="viridis", vmin=-1, vmax=1)
    fig.colorbar(cax)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns)
    ax.set_title(f"{name} Correlation Matrix")
    save_fig(fig, f"{name}_corr.png", folder)

    corr.to_csv(os.path.join(folder, f"{name}_corr.csv"))

corr_heatmap(df1_no_dt, OUTPUT_DIR_1, "dataset_1")
corr_heatmap(df2_no_dt, OUTPUT_DIR_2, "dataset_2")

# ---------- SCATTER ----------
def scatter_pairs(df, folder, name, max_pairs=6):
    cols = df.select_dtypes(include=np.number).columns
    pairs = [(cols[i], cols[i+1]) for i in range(min(len(cols)-1, max_pairs))]
    for x, y in pairs:
        fig, ax = plt.subplots()
        ax.scatter(df[x], df[y], s=10)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(f"{x} vs {y}")
        save_fig(fig, f"{name}_scatter_{x}_vs_{y}.png", folder)

scatter_pairs(df1_no_dt, OUTPUT_DIR_1, "dataset_1")
scatter_pairs(df2_no_dt, OUTPUT_DIR_2, "dataset_2")

# ---------- MEAN COMPARISON ----------
def compare_means(df1, df2, folder):
    cols = list(set(df1.select_dtypes(include=np.number).columns) &
                set(df2.select_dtypes(include=np.number).columns))
    if not cols:
        return

    m1, m2 = df1[cols].mean(), df2[cols].mean()
    comp = pd.DataFrame({"dataset_1": m1, "dataset_2": m2})
    comp.to_csv(os.path.join(folder, "mean_comparison.csv"))

    fig, ax = plt.subplots(figsize=(10,5))
    indices = np.arange(len(cols))
    ax.bar(indices - 0.2, m1, width=0.4, label="dataset_1")
    ax.bar(indices + 0.2, m2, width=0.4, label="dataset_2")
    ax.set_xticks(indices)
    ax.set_xticklabels(cols, rotation=45)
    ax.legend()
    save_fig(fig, "dataset1_vs_dataset2_means.png", folder)

compare_means(df1_no_dt, df2_no_dt, OUTPUT_DIR_1)

# ---------- RANDOM FOREST ----------
def rf_feature_importance(df, target_col, folder, name):
    if target_col not in df.columns:
        print(f"Target missing: {target_col}")
        return

    df = df.dropna(subset=[target_col])
    y = df[target_col].astype(int)

    X = df.drop(columns=[target_col])
    X = pd.get_dummies(X, drop_first=True).fillna(0)

    if X.shape[1] == 0:
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y if y.nunique()>1 else None
    )

    rf = RandomForestClassifier(n_estimators=120, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    fi = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    fi.head(20).to_csv(os.path.join(folder, f"{name}_rf_top_features.csv"))

    fig, ax = plt.subplots(figsize=(7,6))
    top = fi.head(15)
    ax.barh(top.index[::-1], top.values[::-1])
    ax.set_title(f"{name} RF Feature Importance (acc={acc:.2f})")
    save_fig(fig, f"{name}_rf_importance.png", folder)

    print(f"{name} RF Accuracy: {acc:.3f}")

rf_feature_importance(df1_no_dt, TARGET_1, OUTPUT_DIR_1, "dataset_1")
rf_feature_importance(df2_no_dt, TARGET_2, OUTPUT_DIR_2, "dataset_2")

# ---------- PCA + KMEANS ----------
def pca_kmeans_plot(df, folder, name):
    df = df.select_dtypes(include=np.number).dropna()
    if df.shape[1] < 2 or df.shape[0] < 5:
        return

    scaled = StandardScaler().fit_transform(df)

    pcs = PCA(n_components=2).fit_transform(scaled)
    labels = KMeans(n_clusters=3, n_init=10).fit_predict(scaled)

    fig, ax = plt.subplots(figsize=(7,5))
    ax.scatter(pcs[:,0], pcs[:,1], c=labels, s=15)
    ax.set_title(f"{name} PCA + KMeans")
    save_fig(fig, f"{name}_pca_kmeans.png", folder)

pca_kmeans_plot(df1_no_dt, OUTPUT_DIR_1, "dataset_1")
pca_kmeans_plot(df2_no_dt, OUTPUT_DIR_2, "dataset_2")

# ---------- SAVE CLEANED ----------
df1.to_csv(CLEANED_PATH_1, index=False)
df2.to_csv(CLEANED_PATH_2, index=False)

print("\nAll analyses complete.")
print(f"Plots saved to:\n{OUTPUT_DIR_1}\n{OUTPUT_DIR_2}")
print(f"Cleaned files saved to:\n{CLEANED_PATH_1}\n{CLEANED_PATH_2}")
