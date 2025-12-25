import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# page config
st.set_page_config("Logistic Regression", layout="centered")

# Load CSS
def load_css(file):
    with open(file) as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

load_css("style.css")

# Title
st.markdown("""
    <div class="card">
    <h1> Logistic Regression </h1>
    <p> Predict <b>Breast Cancer Diagnosis</b> using medical features </p>
    </div>
""", unsafe_allow_html=True)

# Load Data
@st.cache_data
def load_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    return df, data

df, data = load_data()

# Dataset Preview
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Dataset Preview")
st.dataframe(df.head())
st.markdown('</div>', unsafe_allow_html=True)

# Prepare Data
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Model
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Confusion Matrix
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Confusion Matrix")

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

st.markdown('</div>', unsafe_allow_html=True)

# Performance
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Model Performance")

c1, c2 = st.columns(2)
c1.metric("Accuracy", f"{accuracy:.3f}")
c2.metric("Total Samples", f"{len(df)}")

st.markdown('</div>', unsafe_allow_html=True)

# Classification Report
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Classification Report")

report = classification_report(
    y_test, y_pred,
    target_names=["Malignant", "Benign"],
    output_dict=True
)

st.dataframe(pd.DataFrame(report).transpose())
st.markdown('</div>', unsafe_allow_html=True)

# Prediction
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Predict Diagnosis")

mean_radius = st.slider(
    "Mean Radius",
    float(df["mean radius"].min()),
    float(df["mean radius"].max()),
    float(df["mean radius"].mean())
)

mean_texture = st.slider(
    "Mean Texture",
    float(df["mean texture"].min()),
    float(df["mean texture"].max()),
    float(df["mean texture"].mean())
)

mean_perimeter = st.slider(
    "Mean Perimeter",
    float(df["mean perimeter"].min()),
    float(df["mean perimeter"].max()),
    float(df["mean perimeter"].mean())
)

input_data = np.zeros(X.shape[1])
input_data[0] = mean_radius
input_data[1] = mean_texture
input_data[2] = mean_perimeter

input_scaled = scaler.transform([input_data])
prediction = model.predict(input_scaled)[0]
probability = model.predict_proba(input_scaled)[0][prediction]

result = "Benign" if prediction == 1 else "Malignant"

st.markdown(
    f'<div class="prediction-box">Prediction: {result}<br>Probability: {probability:.2f}</div>',
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)
