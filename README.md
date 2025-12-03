# Sports Injury Prediction

## Objective
Analyze sports-related datasets to predict injury risk and understand performance factors. Motivation: personal experience as a professional badminton athlete where improper rest led to injury. Explore how fatigue, workload, training, and physiological metrics affect injury risk.

---

## Datasets

**1. College Sports Injury Detection**  
- Demographics, training metrics, performance scores, injury labels  
- 200 rows × 17 columns  
- Includes both subjective and objective measurements  

**2. Athlete Injury and Performance Dataset**  
- Session-level physiological and activity metrics, injury risk  
- 1,000+ rows × 13 columns  
- Includes heart rate, respiratory rate, impact force, cumulative fatigue index  

---

## Data Cleaning
- Loaded datasets using `pandas`; inspected structure and summary statistics  
- Checked for missing values, duplicates, and correct data types  
- Created additional metrics (e.g., BMI)  
- Outlier detection using IQR (Dataset 2)  

---

## Analysis & Key Insights
- **Training Intensity:** Slight positive effect on injury risk (Logistic Regression)  
- **Fatigue:** Significant positive correlation with injury (Pearson r = 0.292, p < 0.0001)  
- **Recovery Days:** Fewer recovery days increase injury rates  
- **Activity Type:** Sprinting, Dribbling, Jumping have highest injury risk  
- **Heart Rate & Impact Force:** Minimal predictive effect  
- **Body Composition:** Small influence on fatigue  
- **Weekly Exposure:** Increases injury probability  

Visualizations: logistic regression plots, scatter plots, bar charts, box plots, heatmaps.

---

## Methods
**Statistical:** Logistic Regression, Linear Regression, Pearson Correlation, Chi-Square Test, PCA, KMeans, Group Comparisons  
**Machine Learning:** Random Forest (Dataset 1 Accuracy: 0.95, Dataset 2 Accuracy: 0.603), PCA + KMeans for unsupervised patterns  

---

## Tools
Python 3.12, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`  
Jupyter Notebook / VS Code | macOS  

---

## Workflow
1. Load & inspect datasets  
2. Clean & wrangle data  
3. Exploratory data analysis and visualization  
4. Apply statistical and machine learning models  
5. Interpret results and extract insights for injury prediction and prevention
