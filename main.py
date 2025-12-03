import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

df1 = pd.read_csv("/Users/sudhir/Desktop/Fall2025/Patient_Data/Assignment/Assignment3/Datasets/collegiate_athlete_injury_dataset.csv")
df2 = pd.read_csv("/Users/sudhir/Desktop/Fall2025/Patient_Data/Assignment/Assignment3/Datasets/sports_injury_detection_dataset.csv")

# Standardize IDs
df1["Athlete_ID"] = df1["Athlete_ID"].str.replace("A", "").astype(int)
df2["Athlete_ID"] = df2["Athlete_ID"].str.replace("A", "").astype(int)

# Merge
df = pd.merge(df1, df2, on="Athlete_ID", how="left")

# Feature engineering
df["Training_Load"] = df["Training_Intensity"] * df["Training_Hours_Per_Week"]
df["Weekly_Exposure"] = df["Training_Hours_Per_Week"] + df["Match_Count_Per_Week"]

# Fatigue category
df["Fatigue_Category"] = pd.cut(df["Fatigue_Score"],
                                bins=[0,3,6,10],
                                labels=["Low","Moderate","High"])

# Encode categorical variables
le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])
df["Position"] = le.fit_transform(df["Position"])
df["Activity_Type"] = le.fit_transform(df["Activity_Type"])

# Logistic regression: training intensity → injury
X = df[["Training_Intensity","Training_Load","Weekly_Exposure"]].fillna(0)
y = df["Injury_Indicator"]

model = LogisticRegression()
model.fit(X, y)

print("Logistic Regression Coefficients:")
print(model.coef_)

# Visualization example: fatigue vs injury
sns.boxplot(x=df["Injury_Indicator"], y=df["Fatigue_Score"])
plt.title("Injury vs Fatigue Score")
plt.show()
