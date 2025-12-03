# Advanced_analysis.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import pearsonr, chi2_contingency
import warnings
warnings.filterwarnings("ignore")

# ---------- CONFIG ----------
DATASET_1_PATH = "/Users/sudhir/Desktop/Fall2025/Patient_Data/Assignment/Assignment3/Datasets/collegiate_athlete_injury_dataset_cleaned.csv"
DATASET_2_PATH = "/Users/sudhir/Desktop/Fall2025/Patient_Data/Assignment/Assignment3/Datasets/sports_injury_detection_dataset_cleaned.csv"

OUT_ROOT = "/Users/sudhir/Desktop/Fall2025/Patient_Data/Assignment/Assignment3/Advanced_plots"
OUT1 = os.path.join(OUT_ROOT, "dataset_1")
OUT2 = os.path.join(OUT_ROOT, "dataset_2")
os.makedirs(OUT1, exist_ok=True)
os.makedirs(OUT2, exist_ok=True)

TARGET_1 = "Injury_Indicator"
TARGET_2 = "Injury_Occurred"

# ---------- HELPERS ----------
def save_fig(fig, fname, folder):
    path = os.path.join(folder, fname)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

def write_summary(folder, text):
    path = os.path.join(folder, "results_summary.txt")
    with open(path, "a") as f:
        f.write(text + "\n")
    print(text)

def safe_read(path):
    return pd.read_csv(path)

# ---------- LOAD DATA ----------
df1 = safe_read(DATASET_1_PATH)
df2 = safe_read(DATASET_2_PATH)

# Parse session date if present
for df in (df1, df2):
    if "Session_Date" in df.columns:
        df["Session_Date"] = pd.to_datetime(df["Session_Date"], errors="coerce")

# Drop pure-datetime columns for ML (we will use Session_Date only for time trends)
def drop_datetime_cols(df):
    dt_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64"]).columns.tolist()
    return df.drop(columns=dt_cols, errors="ignore")

df1_no_dt = drop_datetime_cols(df1)
df2_no_dt = drop_datetime_cols(df2)

# ---------- CREATE DERIVED VARIABLES ----------
# Dataset1: Training_Load, Weekly_Exposure, BMI (if height/weight present)
if "Training_Hours_Per_Week" in df1.columns and "Training_Intensity" in df1.columns:
    df1["Training_Load"] = df1["Training_Intensity"] * df1["Training_Hours_Per_Week"]
if "Training_Hours_Per_Week" in df1.columns and "Match_Count_Per_Week" in df1.columns:
    df1["Weekly_Exposure"] = df1["Training_Hours_Per_Week"] + df1["Match_Count_Per_Week"]
if "Height_cm" in df1.columns and "Weight_kg" in df1.columns:
    df1["BMI"] = (df1["Weight_kg"] / ((df1["Height_cm"] / 100) ** 2)).round(1)

# Dataset2: Training_Load_Index may already exist; if not create simple proxy
if "Duration_Minutes" in df2.columns and "Impact_Force_Newtons" in df2.columns:
    df2["Training_Load"] = (df2["Duration_Minutes"] * df2["Impact_Force_Newtons"]).round(1)

# ---------- FUNCTION: RUN LOGISTIC REGRESSION (single predictor) ----------
def run_logistic_single(X_series, y_series, out_folder, plot_name_prefix, xlabel):
    results = {}
    df = pd.DataFrame({ 'X': X_series, 'y': y_series }).dropna()
    if df.empty or df['X'].nunique() < 2 or df['y'].nunique() < 2:
        results['note'] = "Insufficient data"
        return results
    X = df[['X']].values
    y = df['y'].values
    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y if len(np.unique(y))>1 else None)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    prob = model.predict_proba(X)[:,1]
    auc = roc_auc_score(y, prob) if len(np.unique(y))>1 else np.nan
    acc = model.score(X_test, y_test)
    coef = model.coef_[0][0]
    intercept = model.intercept_[0]
    results.update({'coef': float(coef), 'intercept': float(intercept), 'auc': float(auc), 'accuracy': float(acc)})

    # plot scatter + logistic curve
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(df['X'], df['y'], alpha=0.4, s=15)
    # sort X for line
    xs = np.linspace(df['X'].min(), df['X'].max(), 200).reshape(-1,1)
    probs = model.predict_proba(xs)[:,1]
    ax.plot(xs, probs, color='red', linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Injury Probability')
    ax.set_title(f"{plot_name_prefix} logistic fit (acc={acc:.2f}, auc={auc:.2f})")
    save_fig(fig, f"{plot_name_prefix}_logistic.png", out_folder)

    return results

# ---------- FUNCTION: PEARSON + BOXPLOT ----------
def run_pearson_and_box(x, y, out_folder, name):
    df = pd.DataFrame({ 'x': x, 'y': y }).dropna()
    res = {}
    if df.empty or df['x'].nunique() < 2 or df['y'].nunique() < 2:
        res['note'] = "Insufficient data"
        return res
    r, p = pearsonr(df['x'], df['y'])
    res.update({'pearson_r': float(r), 'p_value': float(p)})
    fig, ax = plt.subplots(figsize=(6,4))
    ax.boxplot([df[df['y']==0]['x'].dropna(), df[df['y']==1]['x'].dropna()], labels=['No Injury','Injury'])
    ax.set_title(f"{name} by Injury (r={r:.2f}, p={p:.3f})")
    ax.set_ylabel(name)
    save_fig(fig, f"{name}_box_by_injury.png", out_folder)
    return res

# ---------- FUNCTION: BAR CHART OF GROUP MEANS ----------
def run_group_mean_bar(group_col, target_col, df, out_folder, title_prefix):
    if group_col not in df.columns or target_col not in df.columns:
        return "missing"
    grp = df.groupby(group_col)[target_col].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(grp.index.astype(str), grp.values)
    plt.xticks(rotation=45, ha='right')
    ax.set_ylabel('Mean Injury Rate')
    ax.set_title(f"{title_prefix}: Injury rate by {group_col}")
    save_fig(fig, f"{title_prefix}_injury_by_{group_col}.png", out_folder)
    return grp

# ---------- FUNCTION: CHI-SQUARE ----------
def run_chi_square(cat_col, target_col, df, out_folder):
    if cat_col not in df.columns or target_col not in df.columns:
        return {'note': 'missing'}
    ct = pd.crosstab(df[cat_col], df[target_col])
    chi2, p, dof, ex = chi2_contingency(ct)
    text = f"Chi-square test for {cat_col} vs {target_col}: chi2={chi2:.3f}, p={p:.4f}, dof={dof}"
    write_summary(out_folder, text)
    # save crosstab CSV
    ct.to_csv(os.path.join(out_folder, f"crosstab_{cat_col}_vs_{target_col}.csv"))
    return {'chi2': chi2, 'p': p, 'dof': dof}

# ---------- FUNCTION: MULTIPLE REGRESSION ----------
def run_multiple_regression(X_cols, y_col, df, out_folder, name):
    missing_cols = [c for c in X_cols + [y_col] if c not in df.columns]
    if missing_cols:
        return {'note': f'missing cols {missing_cols}'}
    sub = df[X_cols + [y_col]].dropna()
    if sub.shape[0] < 10:
        return {'note': 'insufficient rows'}
    X = sub[X_cols].values
    y = sub[y_col].values
    model = LinearRegression().fit(X, y)
    coefs = dict(zip(X_cols, model.coef_.tolist()))
    intercept = model.intercept_
    # save a simple summary
    text = f"{name} multiple regression results: intercept={intercept:.3f}, coefs={coefs}"
    write_summary(out_folder, text)
    return {'intercept': float(intercept), 'coefs': coefs}

# ---------- FUNCTION: RANDOM FOREST FEATURE IMPORTANCE ----------
def run_random_forest(df, target_col, out_folder, name):
    if target_col not in df.columns:
        return {'note': 'target missing'}
    dfm = df.dropna(subset=[target_col]).copy()
    X = dfm.drop(columns=[target_col])
    # encode categoricals
    X_enc = pd.get_dummies(X, drop_first=True).fillna(0)
    y = dfm[target_col].astype(int)
    if X_enc.shape[1] < 1 or y.nunique() < 2:
        return {'note': 'no features or single-class target'}
    X_train, X_test, y_train, y_test = train_test_split(X_enc, y, test_size=0.3, random_state=42, stratify=y if y.nunique()>1 else None)
    rf = RandomForestClassifier(n_estimators=150, random_state=42)
    rf.fit(X_train, y_train)
    acc = accuracy_score(y_test, rf.predict(X_test))
    fi = pd.Series(rf.feature_importances_, index=X_enc.columns).sort_values(ascending=False)
    fi.head(20).to_csv(os.path.join(out_folder, f"{name}_rf_feature_importance_top20.csv"))
    fig, ax = plt.subplots(figsize=(8,6))
    top = fi.head(15)
    ax.barh(top.index[::-1], top.values[::-1])
    ax.set_title(f"{name} RF top features (acc={acc:.2f})")
    save_fig(fig, f"{name}_rf_feature_importance.png", out_folder)
    return {'accuracy': float(acc), 'top_features': fi.head(20).to_dict()}

# ---------- FUNCTION: PCA + KMeans ----------
def run_pca_kmeans(df, out_folder, name):
    num = df.select_dtypes(include=np.number).dropna()
    if num.shape[0] < 10 or num.shape[1] < 2:
        return {'note': 'insufficient numeric data'}
    scaler = StandardScaler()
    scaled = scaler.fit_transform(num)
    pcs = PCA(n_components=2).fit_transform(scaled)
    labels = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(scaled)
    fig, ax = plt.subplots(figsize=(7,5))
    ax.scatter(pcs[:,0], pcs[:,1], c=labels, s=15)
    ax.set_title(f"{name} PCA + KMeans")
    save_fig(fig, f"{name}_pca_kmeans.png", out_folder)
    return {'note': 'pca_kmeans_done'}

# -------------------------
# START: Dataset 1 analyses
# -------------------------
write_summary(OUT1, "=== Dataset 1: Advanced Analysis Summary ===")

# Q1: Training Intensity -> Injury (logistic)
if "Training_Intensity" in df1.columns and TARGET_1 in df1.columns:
    r = run_logistic_single(df1["Training_Intensity"], df1[TARGET_1], OUT1, "Q1_training_intensity", "Training Intensity")
    write_summary(OUT1, f"Q1 logistic: {r}")

# Q2: Accumulated fatigue correlate with injury risk (pearson + box)
if "Fatigue_Score" in df1.columns and TARGET_1 in df1.columns:
    r = run_pearson_and_box(df1["Fatigue_Score"], df1[TARGET_1], OUT1, "Fatigue_Score")
    write_summary(OUT1, f"Q2 Pearson (Fatigue vs Injury): {r}")

# Q3: Lower recovery days -> more injuries (bar/group mean)
if "Recovery_Days_Per_Week" in df1.columns and TARGET_1 in df1.columns:
    grp = run_group_mean_bar("Recovery_Days_Per_Week", TARGET_1, df1, OUT1, "Q3")
    write_summary(OUT1, f"Q3 group means by Recovery_Days_Per_Week:\n{grp}")

# Q6: Training load correlate with fatigue (scatter + linear regression)
if "Training_Load" in df1.columns and "Fatigue_Score" in df1.columns:
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(df1["Training_Load"], df1["Fatigue_Score"], s=10, alpha=0.6)
    ax.set_xlabel("Training_Load")
    ax.set_ylabel("Fatigue_Score")
    ax.set_title("Q6 Training_Load vs Fatigue_Score")
    save_fig(fig, "Q6_trainingload_vs_fatigue.png", OUT1)
    lr_res = run_multiple_regression(["Training_Load"], "Fatigue_Score", df1, OUT1, "Q6_TrainingLoad_vs_Fatigue")
    write_summary(OUT1, f"Q6 regression: {lr_res}")

# Q7: Positions more injury-prone (chi-square)
if "Position" in df1.columns and TARGET_1 in df1.columns:
    chi_res = run_chi_square("Position", TARGET_1, df1, OUT1)
    write_summary(OUT1, f"Q7 chi-square: {chi_res}")

# Q9: Body composition influence fatigue (multiple regression)
if "Height_cm" in df1.columns and "Weight_kg" in df1.columns and "Fatigue_Score" in df1.columns:
    res9 = run_multiple_regression(["Height_cm","Weight_kg"], "Fatigue_Score", df1, OUT1, "Q9_BodyComp_vs_Fatigue")
    write_summary(OUT1, f"Q9 multiple regression: {res9}")

# Q10: Weekly exposure (hours + matches) -> injury (logistic)
if "Weekly_Exposure" in df1.columns and TARGET_1 in df1.columns:
    r10 = run_logistic_single(df1["Weekly_Exposure"], df1[TARGET_1], OUT1, "Q10_weekly_exposure", "Weekly Exposure")
    write_summary(OUT1, f"Q10 logistic: {r10}")

# Q8 (activity types) and Q4/Q5 are dataset2 relevant; skip for dataset1.

# Random Forest & PCA/KMeans for dataset1
rf1 = run_random_forest(df1_no_dt, TARGET_1, OUT1, "dataset_1")
write_summary(OUT1, f"RandomForest dataset1: {rf1}")
pca1 = run_pca_kmeans(df1_no_dt, OUT1, "dataset_1")
write_summary(OUT1, f"PCA/KMeans dataset1: {pca1}")

# Save class distribution
if TARGET_1 in df1.columns:
    vc1 = df1[TARGET_1].value_counts().to_string()
    write_summary(OUT1, f"Class distribution {TARGET_1}:\n{vc1}")

# -------------------------
# START: Dataset 2 analyses
# -------------------------
write_summary(OUT2, "=== Dataset 2: Advanced Analysis Summary ===")

# Q4: High heart rate associated with injury (logistic + box)
if "Heart_Rate_BPM" in df2.columns and TARGET_2 in df2.columns:
    r4 = run_logistic_single(df2["Heart_Rate_BPM"], df2[TARGET_2], OUT2, "Q4_heart_rate", "Heart Rate (BPM)")
    write_summary(OUT2, f"Q4 logistic: {r4}")

    # also boxplot
    fig, ax = plt.subplots(figsize=(6,4))
    ax.boxplot([df2[df2[TARGET_2]==0]["Heart_Rate_BPM"].dropna(), df2[df2[TARGET_2]==1]["Heart_Rate_BPM"].dropna()], labels=["No","Yes"])
    ax.set_title("Heart Rate by Injury Occurred")
    save_fig(fig, "Q4_heart_rate_box.png", OUT2)

# Q5: Impact force predict injury (pearson + box)
if "Impact_Force_Newtons" in df2.columns and TARGET_2 in df2.columns:
    r5 = run_pearson_and_box(df2["Impact_Force_Newtons"], df2[TARGET_2], OUT2, "Impact_Force_Newtons")
    write_summary(OUT2, f"Q5 Impact force: {r5}")

# Q8: Activity types injury risk (grouped bar)
if "Activity_Type" in df2.columns and TARGET_2 in df2.columns:
    grp_act = run_group_mean_bar("Activity_Type", TARGET_2, df2, OUT2, "Q8")
    write_summary(OUT2, f"Q8 Activity injury rates:\n{grp_act}")

# Q2 (fatigue correlate) on dataset2 if cumulative fatigue exists
if "Cumulative_Fatigue_Index" in df2.columns and TARGET_2 in df2.columns:
    r2b = run_pearson_and_box(df2["Cumulative_Fatigue_Index"], df2[TARGET_2], OUT2, "Cumulative_Fatigue_Index")
    write_summary(OUT2, f"Q2 (dataset2) Pearson: {r2b}")

# Q6: training load correlate with fatigue (dataset2)
if "Training_Load" in df2.columns and "Cumulative_Fatigue_Index" in df2.columns:
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(df2["Training_Load"], df2["Cumulative_Fatigue_Index"], s=10, alpha=0.6)
    ax.set_xlabel("Training_Load")
    ax.set_ylabel("Cumulative_Fatigue_Index")
    ax.set_title("Q6 (dataset2) Training_Load vs Cumulative_Fatigue_Index")
    save_fig(fig, "Q6_dataset2_trainingload_vs_fatigue.png", OUT2)
    lr_res2 = run_multiple_regression(["Training_Load"], "Cumulative_Fatigue_Index", df2, OUT2, "Q6_dataset2")
    write_summary(OUT2, f"Q6 dataset2 regression: {lr_res2}")

# Q7 (positions) not relevant in dataset2 usually; skip if not present.

# Q9: body comp influence fatigue (if height/weight present in df2)
if "Height_cm" in df2.columns and "Weight_kg" in df2.columns and "Cumulative_Fatigue_Index" in df2.columns:
    r9b = run_multiple_regression(["Height_cm","Weight_kg"], "Cumulative_Fatigue_Index", df2, OUT2, "Q9_dataset2")
    write_summary(OUT2, f"Q9 dataset2 regression: {r9b}")

# Q10 weekly exposure on dataset2 if applicable
if "Duration_Minutes" in df2.columns and "Match_Count_Per_Week" in df2.columns and TARGET_2 in df2.columns:
    # create a proxy Weekly_Exposure if not present
    df2["Weekly_Exposure"] = df2["Duration_Minutes"] + df2["Match_Count_Per_Week"]
    r10b = run_logistic_single(df2["Weekly_Exposure"], df2[TARGET_2], OUT2, "Q10_dataset2_weekly_exposure", "Weekly Exposure (proxy)")
    write_summary(OUT2, f"Q10 dataset2 logistic: {r10b}")

# Random Forest & PCA/KMeans dataset2
rf2 = run_random_forest(df2_no_dt, TARGET_2, OUT2, "dataset_2")
write_summary(OUT2, f"RandomForest dataset2: {rf2}")
pca2 = run_pca_kmeans(df2_no_dt, OUT2, "dataset_2")
write_summary(OUT2, f"PCA/KMeans dataset2: {pca2}")

# Save class distribution
if TARGET_2 in df2.columns:
    vc2 = df2[TARGET_2].value_counts().to_string()
    write_summary(OUT2, f"Class distribution {TARGET_2}:\n{vc2}")

# ---------- Save cleaned copies (non-destructive) ----------
df1.to_csv(os.path.join(os.path.dirname(DATASET_1_PATH), "collegiate_athlete_injury_dataset_advanced_cleaned.csv"), index=False)
df2.to_csv(os.path.join(os.path.dirname(DATASET_2_PATH), "sports_injury_detection_dataset_advanced_cleaned.csv"), index=False)

write_summary(OUT1, "=== End of Dataset 1 analysis ===")
write_summary(OUT2, "=== End of Dataset 2 analysis ===")

print("Advanced analysis finished. Plots and summaries saved to:", OUT_ROOT)
