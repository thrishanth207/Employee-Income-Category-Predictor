import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

columns = [
    "age", "workclass", "fnlwgt", "education",
    "education_num", "marital_status", "occupation",
    "relationship", "race", "sex", "capital_gain",
    "capital_loss", "hours_per_week", "native_country",
    "income"
]

# Load the official test dataset
df = pd.read_csv(
    "data/adult.test",
    names=columns,
    sep=",",
    skipinitialspace=True,
    na_values="?"
)

# Remove extra spaces and the test file's header line
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()

df = df[df["age"] != "|1x3 Cross validator"]
df["income"] = df["income"].str.replace(".", "", regex=False)

# Convert income labels to numbers
df["income"] = df["income"].map({
    "<=50K": 0,
    ">50K": 1
})

# Remove rows with missing or invalid values
df = df.dropna()

X_test = df.drop("income", axis=1)
y_test = df["income"].astype(int)

model_files = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "SVM": "svm.joblib"
}

for name, filename in model_files.items():
    path = Path(filename)

    if not path.exists():
        print(f"\n{name}: model file not found.")
        continue

    model = joblib.load(path)
    predictions = model.predict(X_test)

    print(f"\n{'=' * 45}")
    print(name)
    print(f"{'=' * 45}")
    print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["Income <=50K", "Income >50K"],
        zero_division=0
    ))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

print("\nOfficial test evaluation completed!")