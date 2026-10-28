import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Employee Income Predictor",
    page_icon="💼",
    layout="centered"
)

st.title("💼 Employee Income Category Predictor")
st.write(
    "Enter the details below to predict an income category "
    "using a machine-learning model."
)

# Available trained models
model_files = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "Support Vector Machine (SVM)": "svm.joblib"
}

# Select a model
model_name = st.selectbox(
    "Choose a Machine Learning Model",
    list(model_files.keys())
)

# Input fields
st.subheader("Personal and Employment Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age", min_value=17, max_value=100, value=30
    )
    workclass = st.selectbox(
        "Work Class",
        ["Private", "Self-emp-not-inc", "Self-emp-inc",
         "Federal-gov", "Local-gov", "State-gov",
         "Without-pay", "Never-worked"]
    )
    education = st.selectbox(
        "Education",
        ["Bachelors", "Some-college", "11th", "HS-grad",
         "Prof-school", "Assoc-acdm", "Assoc-voc",
         "9th", "7th-8th", "12th", "Masters",
         "1st-4th", "10th", "Doctorate", "5th-6th",
         "Preschool"]
    )
    marital_status = st.selectbox(
        "Marital Status",
        ["Never-married", "Married-civ-spouse",
         "Divorced", "Married-spouse-absent",
         "Separated", "Widowed", "Married-AF-spouse"]
    )
    occupation = st.selectbox(
        "Occupation",
        ["Tech-support", "Craft-repair", "Other-service",
         "Sales", "Exec-managerial", "Prof-specialty",
         "Handlers-cleaners", "Machine-op-inspct",
         "Adm-clerical", "Farming-fishing",
         "Transport-moving", "Priv-house-serv",
         "Protective-serv", "Armed-Forces"]
    )
    relationship = st.selectbox(
        "Relationship",
        ["Wife", "Own-child", "Husband", "Not-in-family",
         "Other-relative", "Unmarried"]
    )
    race = st.selectbox(
        "Race",
        ["White", "Black", "Asian-Pac-Islander",
         "Amer-Indian-Eskimo", "Other"]
    )

with col2:
    sex = st.selectbox("Sex", ["Male", "Female"])
    fnlwgt = st.number_input(
        "Census Weight", min_value=1, value=100000
    )
    education_num = st.number_input(
        "Education Number", min_value=1, max_value=16, value=10
    )
    capital_gain = st.number_input(
        "Capital Gain", min_value=0, value=0
    )
    capital_loss = st.number_input(
        "Capital Loss", min_value=0, value=0
    )
    hours_per_week = st.number_input(
        "Hours Per Week", min_value=1, max_value=100, value=40
    )
    native_country = st.selectbox(
        "Native Country",
        ["United-States", "India", "Mexico", "Philippines",
         "Germany", "Canada", "England", "Puerto-Rico",
         "El-Salvador", "Cuba", "Other"]
    )

st.info(
    "This model uses historical US census data. "
    "Its predictions may be inaccurate and should not be "
    "used to make real employment or financial decisions."
)

if st.button("🔮 Predict Income", use_container_width=True):
    model_path = Path(
        f"{model_files[model_name]}"
    )

    if not model_path.exists():
        st.error(
            f"Model file not found: {model_path}. "
            "Run train_model.py first."
        )
    else:
        model = joblib.load(model_path)

        input_data = pd.DataFrame([{
            "age": age,
            "workclass": workclass,
            "fnlwgt": fnlwgt,
            "education": education,
            "education_num": education_num,
            "marital_status": marital_status,
            "occupation": occupation,
            "relationship": relationship,
            "race": race,
            "sex": sex,
            "capital_gain": capital_gain,
            "capital_loss": capital_loss,
            "hours_per_week": hours_per_week,
            "native_country": native_country
        }])

        prediction = model.predict(input_data)[0]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.success("Predicted category: Income > $50K")
        else:
            st.success("Predicted category: Income <= $50K")

        st.caption(
            "This is a statistical prediction, not a guarantee "
            "of a person's actual income."
        )
        # Model comparison dashboard
st.divider()

st.header("📊 Model Performance Comparison")

comparison_file = Path("model_comparison.csv")

if comparison_file.exists():
    results = pd.read_csv(comparison_file)
    results = results.sort_values("Accuracy", ascending=False)

    st.subheader("Accuracy of Each Model")

    st.bar_chart(
        results.set_index("Model")["Accuracy"]
    )

    st.subheader("Performance Summary")
    st.dataframe(
        results.style.format({"Accuracy": "{:.2%}"}),
        use_container_width=True
    )

    best_model = results.iloc[0]

    st.success(
        f"Best model by accuracy: {best_model['Model']} "
        f"({best_model['Accuracy']:.2%})"
    )
else:
    st.warning(
        "Model comparison file not found. "
        "Please run train_model.py first."
    )