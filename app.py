import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# -----------------------------------
# Page Config
# -----------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# -----------------------------------
# Title
# -----------------------------------
st.markdown(
    "<h1 style='text-align:center;color:#1f77b4;'>📊 Customer Churn Prediction App</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>Predict whether a customer is likely to churn using Logistic Regression</p>",
    unsafe_allow_html=True
)

st.divider()

# -----------------------------------
# Load Dataset (LOCAL FILE)
# -----------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("telco_dataset.csv")

df = load_data()

# -----------------------------------
# Dataset Preview
# -----------------------------------
st.subheader("🔍 Dataset Preview")
st.dataframe(df.head())

# -----------------------------------
# Encode Categorical Columns
# -----------------------------------
df_encoded = df.copy()
le = LabelEncoder()

for col in df_encoded.select_dtypes(include="object").columns:
    df_encoded[col] = le.fit_transform(df_encoded[col])

# -----------------------------------
# Features & Target
# -----------------------------------
X = df_encoded.drop("Churn", axis=1)
y = df_encoded["Churn"]

# -----------------------------------
# Train-Test Split
# -----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------
# Feature Scaling
# -----------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------------
# Train Logistic Regression
# -----------------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# -----------------------------------
# Predictions
# -----------------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# -----------------------------------
# Metrics
# -----------------------------------
cm = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm.ravel()

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# -----------------------------------
# KPI Counters
# -----------------------------------
st.subheader("📈 Model Performance")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy", f"{acc:.2f}")
c2.metric("Precision", f"{prec:.2f}")
c3.metric("Recall", f"{rec:.2f}")
c4.metric("F1 Score", f"{f1:.2f}")

st.divider()

# -----------------------------------
# Churn Count Plot
# -----------------------------------
st.subheader("📊 Churn Distribution")

fig1, ax1 = plt.subplots()
sns.countplot(x=y, palette=["#2ca02c", "#d62728"], ax=ax1)
ax1.set_xticklabels(["No Churn", "Churn"])
ax1.set_xlabel("Churn Status")
ax1.set_ylabel("Customer Count")
st.pyplot(fig1)

# -----------------------------------
# Confusion Matrix Plot
# -----------------------------------
st.subheader("🔢 Confusion Matrix")

fig2, ax2 = plt.subplots()
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Predicted No", "Predicted Yes"],
    yticklabels=["Actual No", "Actual Yes"],
    ax=ax2
)
ax2.set_xlabel("Prediction")
ax2.set_ylabel("Actual")
st.pyplot(fig2)

# -----------------------------------
# Business Interpretation
# -----------------------------------
st.subheader("💼 Business Interpretation")

st.write(f"✅ **Churn customers correctly identified (TP):** {TP}")
st.write(f"❌ **Non-churn customers wrongly flagged (FP):** {FP}")
st.write(f"⚠️ **Missed churn customers (FN):** {FN}")

st.info(
    "In churn prediction, missing a churn customer (False Negative) "
    "is more costly than wrongly flagging a loyal customer."
)

st.divider()


# -----------------------------------
# Footer
# -----------------------------------
st.markdown(
    "<hr><p style='text-align:center;color:gray;'>Built using Streamlit & Logistic Regression</p>",
    unsafe_allow_html=True
)
