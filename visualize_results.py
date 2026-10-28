import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

columns = [
    "age", "workclass", "fnlwgt", "education",
    "education_num", "marital_status", "occupation",
    "relationship", "race", "sex", "capital_gain",
    "capital_loss", "hours_per_week", "native_country",
    "income"
]

# Load test dataset
df = pd.read_csv(
    "data/adult.test",
    names=columns,
    sep=",",
    skipinitialspace=True,
    na_values="?"
)

# Clean text
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()

# Remove non-data rows
df = df[pd.to_numeric(df["age"], errors="coerce").notna()]

# Clean income labels
df["income"] = df["income"].str.replace(".", "", regex=False)

df["income"] = df["income"].map({
    "<=50K": 0,
    ">50K": 1
})

df = df.dropna()

X_test = df.drop(columns=["income"])
y_test = df["income"].astype(int)

models = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "SVM": "svm.joblib"
}

# Create one figure for all three charts
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (name, filename) in zip(axes, models.items()):
    if not Path(filename).exists():
        ax.set_title(f"{name}\nModel not found")
        ax.axis("off")
        continue

    model = joblib.load(filename)
    predictions = model.predict(X_test)

    cm = confusion_matrix(y_test, predictions)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["<=50K", ">50K"]
    )

    display.plot(ax=ax, colorbar=False)
    ax.set_title(name)
    ax.tick_params(axis="x", rotation=15)

plt.suptitle("Income Prediction: Model Comparison", fontsize=16)
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=300, bbox_inches="tight")
plt.show()

print("All confusion matrices displayed!")
print("Chart saved as confusion_matrices.png")