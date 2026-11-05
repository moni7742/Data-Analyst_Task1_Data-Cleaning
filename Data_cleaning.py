
import pandas as pd
import numpy as np
import os

# Create output folder
os.makedirs("output", exist_ok=True)

# 1. Load dataset
df = pd.read_csv("dataset/marketing_campaign.csv", sep="\t")

print("ORIGINAL DATASET")
print("Shape:", df.shape)
print(df.head())

original_rows = len(df)
original_columns = len(df.columns)

# 2. Standardize column names
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

# 3. Check missing values
print("\nMissing values before cleaning:")
print(df.isnull().sum())

missing_before = int(df.isnull().sum().sum())

# 4. Remove duplicate records
duplicates_before = int(df.duplicated().sum())
df = df.drop_duplicates().copy()

# 5. Standardize text columns
text_columns = df.select_dtypes(include="object").columns

for col in text_columns:
    df[col] = df[col].str.strip()

# Standardize education values
df["education"] = df["education"].str.title()

# Standardize marital status values
df["marital_status"] = df["marital_status"].str.title()

df["marital_status"] = df["marital_status"].replace({
    "Alone": "Single",
    "Absurd": "Other",
    "Yolo": "Other"
})

# 6. Handle missing numerical values
# Use median for Income because it is less affected by outliers
income_median = df["income"].median()
df["income"] = df["income"].fillna(income_median)

# 7. Convert date column into datetime
df["dt_customer"] = pd.to_datetime(
    df["dt_customer"],
    dayfirst=True,
    errors="coerce"
)

# 8. Check data types
print("\nData types after cleaning:")
print(df.dtypes)

# 9. Check missing values after cleaning
missing_after = int(df.isnull().sum().sum())

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# 10. Check outliers using IQR method
numeric_columns = df.select_dtypes(include=np.number).columns

outlier_counts = {}

for col in numeric_columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    count = int(
        ((df[col] < lower_limit) |
         (df[col] > upper_limit)).sum()
    )

    if count > 0:
        outlier_counts[col] = count

print("\nPotential outliers:")
print(outlier_counts)

# 11. Save cleaned dataset
df.to_csv("output/cleaned_customer_data.csv", index=False)

# 12. Generate cleaning summary
summary = f"""
DATA CLEANING AND PREPROCESSING REPORT

Dataset: Customer Personality Analysis

Original number of rows: {original_rows}
Original number of columns: {original_columns}

Duplicate records identified: {duplicates_before}
Rows after duplicate removal: {len(df)}

Missing values before cleaning: {missing_before}
Missing values after cleaning: {missing_after}

Income missing values were filled using the median.

Text values were standardized.
Column names were converted to lowercase.
Date column was converted into datetime format.
Potential outliers were identified using the IQR method.

Final number of rows: {len(df)}
Final number of columns: {len(df.columns)}

Output file: cleaned_customer_data.csv

The cleaned dataset is ready for further analysis.
"""

with open("output/cleaning_summary.txt", "w") as file:
    file.write(summary)

print("\n", summary)
print("DATA CLEANING COMPLETED SUCCESSFULLY!")
