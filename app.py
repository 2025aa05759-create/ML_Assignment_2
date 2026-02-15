import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef, confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# -------------------------------
# App Title & Branding
# -------------------------------
st.set_page_config(page_title="ML Assignment 2 - Ranald Nivethan X N", layout="wide")
st.title("📊 Machine Learning Assignment 2")
st.markdown("**Author:** Ranald Nivethan X N  \n**Course:** M.Tech (AIML) - BITS Pilani")
st.markdown("---")

# -------------------------------
# Sidebar - Upload dataset
# -------------------------------
st.sidebar.header("Upload & Configure")
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV)", type="csv")

if uploaded_file:
    
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    st.subheader(f"length of dataframe: {len(df)}")

    # -------------------------------
    # Target selection
    # -------------------------------
    target_col = st.sidebar.selectbox("Select target column", df.columns)

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Encode target if categorical
    if y.dtype == "object" or y.dtype.name == "category":
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # -------------------------------
    # Preprocessing: numeric + categorical
    # -------------------------------
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns
    numeric_cols = X.select_dtypes(include=["int64","float64"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )

    # -------------------------------
    # Model selection
    # -------------------------------
    st.sidebar.subheader("Select Model")
    model_choice = st.sidebar.selectbox(
        "Choose a classifier",
        ["Logistic Regression", "Decision Tree", "kNN", "Naive Bayes", "Random Forest", "XGBoost"]
    )

    if st.sidebar.button("Run Model"):
        if model_choice == "Logistic Regression":
            classifier = LogisticRegression(max_iter=1000)
        elif model_choice == "Decision Tree":
            classifier = DecisionTreeClassifier()
        elif model_choice == "kNN":
            classifier = KNeighborsClassifier()
        elif model_choice == "Naive Bayes":
            classifier = GaussianNB()
        elif model_choice == "Random Forest":
            classifier = RandomForestClassifier(n_estimators=100)
        else:
            classifier = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                eval_metric="logloss",
                verbosity=0
            )

        # Build pipeline
        model = Pipeline(steps=[("preprocessor", preprocessor),
                                ("classifier", classifier)])

        # Train & predict
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Try probabilities for AUC
        y_prob = None
        if hasattr(model, "predict_proba"):
            try:
                y_prob = model.predict_proba(X_test)
            except Exception:
                y_prob = None

        # Detect binary vs multi-class
        is_multiclass = len(np.unique(y_test)) > 2

        # Metrics
        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average="weighted" if is_multiclass else "binary"),
            "Recall": recall_score(y_test, y_pred, average="weighted" if is_multiclass else "binary"),
            "F1": f1_score(y_test, y_pred, average="weighted" if is_multiclass else "binary"),
            "MCC": matthews_corrcoef(y_test, y_pred)
        }

        if y_prob is not None:
            try:
                if is_multiclass:
                    metrics["AUC"] = roc_auc_score(y_test, y_prob, multi_class="ovr")
                else:
                    metrics["AUC"] = roc_auc_score(y_test, y_prob[:, 1])
            except Exception as e:
                metrics["AUC"] = f"Error computing AUC: {e}"

        # -------------------------------
        # Display results
        # -------------------------------
        st.subheader("📈 Evaluation Metrics")
        for metric, value in metrics.items():
            st.write(f"**{metric}:** {value}")

        st.subheader("🔎 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

else:
    st.info("⬅️ Please upload a dataset (CSV) from the sidebar to begin.")