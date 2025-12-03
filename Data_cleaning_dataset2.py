import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Configure pandas display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_colwidth', None)

# Load the dataset
file_path = "/Users/sudhir/Desktop/Fall2025/Patient_Data/Assignment/Assignment3/Datasets/sports_injury_detection_dataset.csv"
df2 = pd.read_csv(file_path)

# ======================================
# HEAD (TRANSPOSED)
# ======================================
print("======================================")
print("DATASET 2: HEAD OF DATASET (TRANSPOSED)")
print("======================================")
head_transposed = df2.head().T
head_transposed.index.name = "Feature"
print(head_transposed, "\n")

# ======================================
# BASIC INFO
# ======================================
print("======================================")
print("DATASET 2: BASIC INFORMATION")
print("======================================")
print(df2.info(), "\n")

# ======================================
# SUMMARY STATISTICS
# ======================================
print("======================================")
print("DATASET 2: SUMMARY STATISTICS (TRANSPOSED)")
print("======================================")
summary_stats = df2.describe(include="all").T
summary_stats.index.name = "Feature"
print(summary_stats, "\n")

# ======================================
# NULL VALUES
# ======================================
print("======================================")
print("NULL VALUES PER COLUMN")
print("======================================")
print(df2.isnull().sum(), "\n")

# ======================================
# DUPLICATE ROW CHECK
# ======================================
print("======================================")
print("NUMBER OF DUPLICATE ROWS")
print("======================================")
duplicate_count = df2.duplicated().sum()
print(f"Duplicate rows: {duplicate_count}\n")

if duplicate_count > 0:
    print("Duplicate rows detail:")
    print(df2[df2.duplicated()], "\n")

# ======================================
# DATA TYPE VALIDATION
# ======================================
print("======================================")
print("DATA TYPE VALIDATION")
print("======================================")
expected_types2 = {
    "Athlete_ID": "object",
    "Sport_Type": "object",
    "Session_Date": "object",
    "Heart_Rate_BPM": "int64",
    "Respiratory_Rate_BPM": "int64",
    "Skin_Temperature_C": "float64",
    "Blood_Oxygen_Level_Percent": "float64",
    "Impact_Force_Newtons": "float64",
    "Cumulative_Fatigue_Index": "float64",
    "Activity_Type": "object",
    "Duration_Minutes": "int64",
    "Injury_Risk_Score": "float64",
    "Injury_Occurred": "int64"
}

for col, expected in expected_types2.items():
    if col in df2.columns:
        if df2[col].dtype != expected:
            print(f"⚠️ Column '{col}' has type {df2[col].dtype} but expected {expected}")
        else:
            print(f"✔ Column '{col}' matches expected type: {expected}")

# ======================================
# CREATE NEW VARIABLES
# ======================================
print("\n======================================")
print("CREATING NEW VARIABLES")
print("======================================")

df2["Training_Load_Index"] = df2["Duration_Minutes"] * df2["Impact_Force_Newtons"]
df2["Fatigue_to_Recovery_Index"] = df2["Cumulative_Fatigue_Index"] / df2["Duration_Minutes"].replace(0, np.nan)
df2["Oxygen_Efficiency_Ratio"] = df2["Blood_Oxygen_Level_Percent"] / df2["Heart_Rate_BPM"]
df2["High_Risk_Session"] = (df2["Injury_Risk_Score"] > 0.7).astype(int)
df2["Stress_Index"] = df2["Cumulative_Fatigue_Index"] * df2["Heart_Rate_BPM"]

# Round decimal-heavy columns
cols_to_round = [
    "Skin_Temperature_C",
    "Blood_Oxygen_Level_Percent",
    "Impact_Force_Newtons",
    "Cumulative_Fatigue_Index",
    "Injury_Risk_Score",
    "Training_Load_Index",
    "Fatigue_to_Recovery_Index",
    "Oxygen_Efficiency_Ratio",
    "Stress_Index",
]

df2[cols_to_round] = df2[cols_to_round].round(2)

print("New variables created and rounded successfully.\n")

# ======================================
# HEAD WITH NEW VARIABLES (TRANSPOSED)
# ======================================
print("======================================")
print("HEAD OF DATASET WITH NEW VARIABLES (TRANSPOSED)")
print("======================================")
head_transposed_new = df2.head().T
head_transposed_new.index.name = "Feature"
print(head_transposed_new, "\n")

# ======================================
# VISUALIZATIONS
# ======================================
print("======================================")
print("GENERATING VISUALIZATIONS")
print("======================================")

# 1. Heart Rate vs Injury Occurrence
sns.boxplot(x="Injury_Occurred", y="Heart_Rate_BPM", data=df2)
plt.title("Heart Rate vs Injury Occurrence")
plt.show()

# 2. Impact Force vs Injury Risk Score
plt.scatter(df2["Impact_Force_Newtons"], df2["Injury_Risk_Score"])
plt.xlabel("Impact Force (Newtons)")
plt.ylabel("Injury Risk Score")
plt.title("Impact Force vs Injury Risk Score")
plt.show()

# 3. Injury Rate by Activity Type
inj_rate = df2.groupby("Activity_Type")["Injury_Occurred"].mean()
inj_rate.plot(kind="bar")
plt.title("Injury Rate by Activity Type")
plt.ylabel("Average Injury Occurrence")
plt.show()

# 4. Fatigue Index Distribution
df2["Cumulative_Fatigue_Index"].plot(kind="hist", bins=20)
plt.title("Fatigue Index Distribution")
plt.show()

# 5. Correlation Heatmap
sns.heatmap(df2.corr(), annot=False, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# 6. Stress Index vs Injury Occurrence
sns.boxplot(x="Injury_Occurred", y="Stress_Index", data=df2)
plt.title("Stress Index vs Injury Occurrence")
plt.show()

# 7. Training Load vs Injury Risk
sns.regplot(x="Training_Load_Index", y="Injury_Risk_Score", data=df2)
plt.title("Training Load vs Injury Risk Score")
plt.show()

# ======================================
# SAVE CLEANED DATASET
# ======================================
cleaned_path2 = "/Users/sudhir/Desktop/Fall2025/Patient_Data/Assignment/Assignment3/Datasets/sports_injury_detection_dataset_cleaned.csv"
df2.to_csv(cleaned_path2, index=False)
print("Cleaned dataset saved successfully at:")
print(cleaned_path2)
