import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_colwidth', None)

file_path = "/Users/sudhir/Desktop/Fall2025/Patient_Data/Assignment/Assignment3/Datasets/collegiate_athlete_injury_dataset.csv"
df = pd.read_csv(file_path)

output_folder = "data_cleaning_visuals"
os.makedirs(output_folder, exist_ok=True)

print("======================================")
print("HEAD OF DATASET (TRANSPOSED)")
print("======================================")
head_transposed = df.head().T
head_transposed.index.name = 'Feature'
print(head_transposed, "\n")

plt.figure(figsize=(12, 6))
plt.title("Missing Values Heatmap (Head of Dataset)")
plt.imshow(df.head().isnull(), aspect='auto')
plt.xlabel("Columns")
plt.ylabel("Rows")
plt.colorbar(label="Missing = 1")
plt.tight_layout()
plt.savefig(f"{output_folder}/missing_values_heatmap_head.png")
plt.close()

print("======================================")
print("BASIC INFORMATION")
print("======================================")
print(df.info(), "\n")

print("======================================")
print("SUMMARY STATISTICS (TRANSPOSED)")
print("======================================")
summary = df.describe(include='all').T
summary.index.name = 'Feature'
print(summary, "\n")

summary_numeric = df.describe().T
plt.figure(figsize=(12, 6))
plt.title("Distribution of Numeric Columns (Boxplot)")
df.boxplot(rot=90)
plt.tight_layout()
plt.savefig(f"{output_folder}/numeric_distribution_boxplot.png")
plt.close()

print("======================================")
print("NULL VALUES PER COLUMN")
print("======================================")
null_counts = df.isnull().sum()
print(null_counts, "\n")

plt.figure(figsize=(12, 6))
plt.title("Null Values Per Column")
null_counts.plot(kind="bar")
plt.tight_layout()
plt.savefig(f"{output_folder}/null_values_barplot.png")
plt.close()

print("======================================")
print("NUMBER OF DUPLICATE ROWS")
print("======================================")
duplicate_count = df.duplicated().sum()
print(duplicate_count, "\n")

if duplicate_count > 0:
    print("Duplicate rows found:")
    print(df[df.duplicated()], "\n")

print("======================================")
print("WRONG OR SUSPICIOUS DATA TYPES")
print("======================================")

expected_types = {
    "Athlete_ID": "object",
    "Age": "int64",
    "Gender": "object",
    "Height_cm": "int64",
    "Weight_kg": "int64",
    "Position": "object",
    "Training_Intensity": "int64",
    "Training_Hours_Per_Week": "int64",
    "Recovery_Days_Per_Week": "int64",
    "Match_Count_Per_Week": "int64",
    "Rest_Between_Events_Days": "int64",
    "Fatigue_Score": "int64",
    "Performance_Score": "int64",
    "Team_Contribution_Score": "int64",
    "Load_Balance_Score": "int64",
    "ACL_Risk_Score": "int64",
    "Injury_Indicator": "int64"
}

dtype_mismatch = {}

for col, expected in expected_types.items():
    if col in df.columns:
        if df[col].dtype != expected:
            dtype_mismatch[col] = str(df[col].dtype)
            print(f"Column '{col}' has type {df[col].dtype} but expected {expected}")
        else:
            print(f"Column '{col}' has the expected type {expected}")

print("\n")

plt.figure(figsize=(10, 5))
plt.title("Data Type Counts")
df.dtypes.value_counts().plot(kind="bar")
plt.tight_layout()
plt.savefig(f"{output_folder}/data_types_barplot.png")
plt.close()

print("======================================")
print("CREATING NEW VARIABLES")
print("======================================")

df["Training_Load_Index"] = df["Training_Hours_Per_Week"] + df["Match_Count_Per_Week"]
df["Fatigue_to_Recovery_Ratio"] = df["Fatigue_Score"] / df["Recovery_Days_Per_Week"].replace(0, np.nan)
df["Weekly_Stress_Score"] = (df["Training_Intensity"] * df["Training_Hours_Per_Week"]) + df["Match_Count_Per_Week"]
df["Height_Weight_Ratio"] = df["Height_cm"] / df["Weight_kg"]
df["High_Risk_Athlete_Indicator"] = (df["ACL_Risk_Score"] >= 7).astype(int)

df.insert(
    loc=df.columns.get_loc("Weight_kg") + 1,
    column="BMI",
    value=(df["Weight_kg"] / ((df["Height_cm"] / 100) ** 2)).round(1)
)

df["Fatigue_to_Recovery_Ratio"] = df["Fatigue_to_Recovery_Ratio"].round(1)
df["Height_Weight_Ratio"] = df["Height_Weight_Ratio"].round(1)

print("New variables including BMI created and rounded successfully.\n")

plt.figure(figsize=(12, 6))
plt.title("Distribution of BMI")
df["BMI"].plot(kind="hist", bins=20)
plt.tight_layout()
plt.savefig(f"{output_folder}/bmi_distribution.png")
plt.close()

print("======================================")
print("HEAD OF DATASET WITH NEW VARIABLES (TRANSPOSED)")
print("======================================")
head_new_transposed = df.head().T
head_new_transposed.index.name = "Feature"
print(head_new_transposed, "\n")

cleaned_path = "/Users/sudhir/Desktop/Fall2025/Patient_Data/Assignment/Assignment3/Datasets/collegiate_athlete_injury_dataset_cleaned.csv"
df.to_csv(cleaned_path, index=False)

print("Cleaned dataset saved successfully at:")
print(cleaned_path)
print("Visuals saved in:", output_folder)
