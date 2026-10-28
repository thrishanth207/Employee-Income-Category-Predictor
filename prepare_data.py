import pandas as pd

columns = [
    "age", "workclass", "fnlwgt", "education",
    "education_num", "marital_status", "occupation",
    "relationship", "race", "sex", "capital_gain",
    "capital_loss", "hours_per_week", "native_country",
    "income"
]

# Load the dataset
df = pd.read_csv(
    "data/adult.data",
    names=columns,
    sep=",",
    skipinitialspace=True,
    na_values="?"
)

# Remove extra spaces from text columns
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()

# Remove rows with missing values
df = df.dropna()

# Convert income into a prediction target
df["income"] = df["income"].map({
    "<=50K": 0,
    ">50K": 1
})

# Remove rows with unrecognized income labels
df = df.dropna(subset=["income"])
df["income"] = df["income"].astype(int)

# Create a cleaned-data folder
import os
os.makedirs("data/processed", exist_ok=True)

# Save the prepared dataset
df.to_csv("data/processed/adult_cleaned.csv", index=False)

print("Dataset preparation completed!")
print("Cleaned dataset shape:", df.shape)
print("\nIncome category counts:")
print(df["income"].value_counts())
print("\nSaved to data/processed/adult_cleaned.csv")