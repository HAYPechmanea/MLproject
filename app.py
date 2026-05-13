# =========================================================
# STREAMLIT WEB APPLICATION
# SVM CUSTOMER CLASSIFICATION
# =========================================================

# Run command:
# streamlit run app.py

# =========================================================
# IMPORT LIBRARIES
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="SVM Customer Classification",
    layout="wide"
)

# =========================================================
# TITLE
# =========================================================

st.title("Support Vector Machine Customer Classification")

st.markdown("""
This web application predicts whether a customer is:

- Safe Customer
- Risky Customer

using different SVM kernels:
- Linear SVM
- Polynomial Kernel SVM
- RBF Kernel SVM
""")

# =========================================================
# LOAD DATASET
# =========================================================

# IMPORTANT:
# Your dataset must contain:
# Income, Debt_Ratio, Class

df = pd.read_csv("xor_dataset.csv")

# =========================================================
# SHOW DATASET
# =========================================================

st.subheader("Dataset Preview")

st.dataframe(df.head())

# =========================================================
# FEATURES AND TARGET
# =========================================================

X = df[['Income', 'Debt_Ratio']]

# Class:
# 0 = Risky Customer
# 1 = Safe Customer

y = df['Class']

# =========================================================
# ORIGINAL DATASET GRAPH
# =========================================================

st.subheader("Original Dataset")

fig1, ax1 = plt.subplots(figsize=(8,6))

# Risky customer

ax1.scatter(
    X[y == 0]['Income'],
    X[y == 0]['Debt_Ratio'],
    label='Risky Customer'
)

# Safe customer

ax1.scatter(
    X[y == 1]['Income'],
    X[y == 1]['Debt_Ratio'],
    label='Safe Customer'
)

ax1.set_xlabel("Income")
ax1.set_ylabel("Debt Ratio (%)")

ax1.set_title("Original Dataset Distribution")

ax1.legend()

ax1.grid(True)

st.pyplot(fig1)

# =========================================================
# SPLIT TRAINING AND TEST DATA
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================================================
# FEATURE SCALING
# =========================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# =========================================================
# TRAIN MODELS
# =========================================================

# Linear SVM

linear_model = SVC(kernel='linear')

linear_model.fit(X_train, y_train)

# Polynomial Kernel SVM

poly_model = SVC(
    kernel='poly',
    degree=2
)

poly_model.fit(X_train, y_train)

# RBF Kernel SVM

rbf_model = SVC(kernel='rbf')

rbf_model.fit(X_train, y_train)

# =========================================================
# SIDEBAR INPUT
# =========================================================

st.sidebar.header("Customer Input")

# Income input

income = st.sidebar.slider(
    "Income ($)",
    0,
    10000,
    5000
)

# Debt ratio input

debt_ratio = st.sidebar.slider(
    "Debt Ratio (%)",
    0,
    100,
    50
)

# Kernel selection

kernel_choice = st.sidebar.selectbox(
    "Choose SVM Kernel",
    (
        "Linear SVM",
        "Polynomial Kernel SVM",
        "RBF Kernel SVM"
    )
)

# =========================================================
# USER INPUT DATA
# =========================================================

input_data = pd.DataFrame({
    'Income': [income],
    'Debt_Ratio': [debt_ratio]
})

# Scale input

input_scaled = scaler.transform(input_data)

# =========================================================
# SELECT MODEL
# =========================================================

if kernel_choice == "Linear SVM":

    model = linear_model

elif kernel_choice == "Polynomial Kernel SVM":

    model = poly_model

else:

    model = rbf_model

# =========================================================
# PREDICTION
# =========================================================

prediction = model.predict(input_scaled)

# =========================================================
# SHOW PREDICTION RESULT
# =========================================================

st.subheader("Prediction Result")

if prediction[0] == 1:

    st.success("Safe Customer")

else:

    st.error("Risky Customer")

# =========================================================
# DECISION BOUNDARY VISUALIZATION
# =========================================================

st.subheader("Decision Boundary Visualization")

# Scale all dataset

X_scaled_all = scaler.transform(X)

# Convert y

y_array = np.array(y)

# Create meshgrid

h = 0.02

x_min = X_scaled_all[:, 0].min() - 1
x_max = X_scaled_all[:, 0].max() + 1

y_min = X_scaled_all[:, 1].min() - 1
y_max = X_scaled_all[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, h),
    np.arange(y_min, y_max, h)
)

# Predict mesh points

Z = model.predict(
    np.c_[xx.ravel(), yy.ravel()]
)

Z = Z.reshape(xx.shape)

# =========================================================
# PLOT GRAPH
# =========================================================

fig2, ax2 = plt.subplots(figsize=(8,6))

# Decision boundary

ax2.contourf(xx, yy, Z, alpha=0.3)

# Risky customer

ax2.scatter(
    X_scaled_all[y_array == 0, 0],
    X_scaled_all[y_array == 0, 1],
    label='Risky Customer'
)

# Safe customer

ax2.scatter(
    X_scaled_all[y_array == 1, 0],
    X_scaled_all[y_array == 1, 1],
    label='Safe Customer'
)

# User point

user_scaled = scaler.transform(input_data)

ax2.scatter(
    user_scaled[:, 0],
    user_scaled[:, 1],
    s=250,
    marker='X',
    label='User Input'
)

ax2.set_title(kernel_choice)

ax2.set_xlabel("Scaled Income")
ax2.set_ylabel("Scaled Debt Ratio")

ax2.legend()

ax2.grid(True)

# Show graph

st.pyplot(fig2)

# =========================================================
# MODEL INFORMATION
# =========================================================

st.subheader("Model Information")

if kernel_choice == "Linear SVM":

    st.write("""
    Linear SVM uses a straight decision boundary.
    It works better for linear datasets.
    """)

elif kernel_choice == "Polynomial Kernel SVM":

    st.write("""
    Polynomial Kernel SVM creates curved nonlinear boundaries.
    It performs better on nonlinear datasets.
    """)

else:

    st.write("""
    RBF Kernel SVM creates flexible nonlinear decision boundaries.
    It is usually the best model for nonlinear classification.
    """)