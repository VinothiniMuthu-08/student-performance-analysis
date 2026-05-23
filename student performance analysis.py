# ============================================
# DATA CLEANING & VISUALIZATION PROJECT
# ============================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------
# STEP 1: LOAD DATASET
# --------------------------------------------

# Replace with your dataset file name
df = pd.read_csv(
    r"C:\Users\VINOTHINI\Documents\student-mat.csv",
    sep=';'
)

# Display first 5 rows
print("FIRST 5 ROWS")
print(df.head())

# --------------------------------------------
# STEP 2: BASIC INFORMATION
# --------------------------------------------

print("\nDATASET INFO")
print(df.info())

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nSTATISTICAL SUMMARY")
print(df.describe())

# --------------------------------------------
# STEP 3: HANDLE MISSING VALUES
# --------------------------------------------

## Fill missing numerical values with mean (safe method)
numeric_columns = df.select_dtypes(include=np.number).columns

for col in numeric_columns:
    df[col] = df[col].fillna(df[col].mean())

# --------------------------------------------
# STEP 4: REMOVE DUPLICATES
# --------------------------------------------

print("\nDUPLICATES BEFORE:", df.duplicated().sum())

df.drop_duplicates(inplace=True)

print("DUPLICATES AFTER:", df.duplicated().sum())

# --------------------------------------------
# STEP 5: DETECT OUTLIERS USING IQR
# --------------------------------------------

def remove_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    return data[(data[column] >= lower) & (data[column] <= upper)]

# Remove outliers from math score
df = remove_outliers(df, 'G3')

# --------------------------------------------
# STEP 6: FEATURE ENGINEERING
# --------------------------------------------

# Create Total Marks Column
df['Total Score'] = (
    df['G1'] +
    df['G2'] +
    df['G3']
)


# Create Average Score Column
df['Average Score'] = df['Total Score'] / 3

# --------------------------------------------
# STEP 7: DATA VISUALIZATION
# --------------------------------------------

# Set style
sns.set(style="whitegrid")

# --------------------------------------------
# CHART 1: GENDER COUNT
# --------------------------------------------

plt.figure(figsize=(6,5))

sns.countplot(x='sex', data=df)

plt.title("Gender Distribution")

plt.show()

# --------------------------------------------
# CHART 2: AVERAGE MATH SCORE BY GENDER
# --------------------------------------------

plt.figure(figsize=(6,5))

sns.barplot(x='sex', y='G3', data=df)

plt.title("Average Math Score by Gender")

plt.show()

# --------------------------------------------
# CHART 3: SCORE DISTRIBUTION
# --------------------------------------------

plt.figure(figsize=(8,5))

sns.histplot(df['Average Score'], bins=20, kde=True)

plt.title("Average Score Distribution")

plt.show()

# --------------------------------------------
# CHART 4: CORRELATION HEATMAP
# --------------------------------------------

plt.figure(figsize=(8,6))

correlation = df.select_dtypes(include=np.number).corr()

sns.heatmap(correlation, annot=True)

plt.title("Correlation Heatmap")

plt.show()

# --------------------------------------------
# CHART 5: PARENT EDUCATION VS SCORE
# --------------------------------------------

plt.figure(figsize=(10,5))

sns.barplot(
    x='studytime',
    y='Average Score',
    data=df
)

plt.xticks(rotation=45)

plt.title("Parent Education vs Average Score")

plt.show()

# --------------------------------------------
# STEP 8: SAVE CLEANED DATA
# --------------------------------------------

df.to_csv("Cleaned_Students_Data.csv", index=False)

print("\nCLEANED DATA SAVED SUCCESSFULLY")

# --------------------------------------------
# STEP 9: FINAL INSIGHTS
# --------------------------------------------

print("\nFINAL INSIGHTS")

print("1. Female and Male student counts analyzed.")

print("2. Average scores calculated.")

print("3. Outliers removed from math scores.")

print("4. Correlation between subjects visualized.")

print("5. Cleaned dataset exported successfully.")
