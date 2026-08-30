



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, roc_curve, roc_auc_score

import io
from google.colab import files
uploaded = files.upload()
dataframe = pd.read_csv(io.StringIO(uploaded['test.csv'].decode('utf-8')))

dataframe

"""Understanding Dataset"""

dataframe.info()

"""Duplicates || Missing Values || Target Variable"""

dataframe.duplicated().sum()
dataframe.isnull().sum()
dataframe['Clicked on Ad'].value_counts()

"""# Let's proceed step by step with our Ad Click Prediction ML project.

# Data Cleaning
"""

print(dataframe.columns)

dataframe = dataframe.rename(
    columns={
        'Clicked on Ad': 'Clicked_on_Ad',
        'Area Income': 'Area_Income'
    }
)

dataframe = dataframe[
    dataframe['Clicked_on_Ad'].isin([0, 1])
].copy()

print(dataframe['Clicked_on_Ad'].value_counts())

# Clean Duration directly in dataframe
dataframe['Duration'] = (
    dataframe['Duration']
    .astype(str)
    .str.extract(r'(\d+)')[0]
    .astype(int)
)

print(dataframe['Duration'].head())
print(dataframe['Duration'].dtype)

dataframe['Date'] = pd.to_datetime(
    dataframe['Date'],
    format='mixed',
    errors='coerce'
)

dataframe['Year'] = dataframe['Date'].dt.year
dataframe['Month'] = dataframe['Date'].dt.month
dataframe['Day'] = dataframe['Date'].dt.day
dataframe['Day_of_Week'] = dataframe['Date'].dt.dayofweek
print(dataframe['Month'])

print(dataframe['Duration'].unique())
dataframe = dataframe.drop(columns=['Date'])

"""# Exploratory Data Analysis

## What are the impressions of the ads of a particular person?
"""

plt.figure(figsize=(10, 7))

sns.histplot(
    data=dataframe,
    x='Impressions',
    bins=50,
    kde=True,
    edgecolor="k",
    linewidth=1
)

plt.show()

"""### Variance Inflation Factor (VIF)."""

numerical_columns = ['Conversion_Rate', 'ROI', 'Clicks', 'Impressions', 'Engagement_Score', 'Gender', 'Age', 'Area_Income' ]

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import pandas as pd

# Convert selected columns into a DataFrame
numerical_data = dataframe[numerical_columns]

def calculate_vif(data):

    # Add constant/intercept
    X = add_constant(data)

    vif_data = pd.DataFrame()

    vif_data["Feature"] = X.columns

    vif_data["VIF"] = [
        variance_inflation_factor(X.values, i)
        for i in range(X.shape[1])
    ]

    return vif_data

vif_data = calculate_vif(numerical_data)

print(vif_data)

correlation_matrix = numerical_data.corr()

plt.figure(figsize=(10, 7))
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='coolwarm',
    fmt='.2f'
)

plt.show()

"""## Separate numerical and categorical columns"""

X = dataframe.drop(columns=['Clicked_on_Ad'])

y = dataframe['Clicked_on_Ad']

numerical_features = X.select_dtypes(
    include=['int64', 'float64']
).columns

categorical_features = X.select_dtypes(
    include=['object']
).columns

print("Numerical Features:")
print(numerical_features)

print("\nCategorical Features:")
print(categorical_features)

"""# Train-test split"""

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

"""# Preprocessing pipeline
We need:

Numerical data → scale using StandardScaler

Categorical data → encode using OneHotEncoder
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

preprocessor = ColumnTransformer(
    transformers=[
        (
            'num',
            StandardScaler(),
            numerical_features
        ),
        (
            'cat',
            OneHotEncoder(handle_unknown='ignore'),
            categorical_features
        )
    ]
)

"""# Dummy Classifier"""

from sklearn.dummy import DummyClassifier

dummy_model = DummyClassifier(
    strategy='most_frequent'
)

dummy_model.fit(X_train, y_train)

dummy_pred = dummy_model.predict(X_test)

print(
    "Baseline Accuracy:",
    accuracy_score(y_test, dummy_pred)
)

"""# Logistic Regression baseline model"""

from sklearn.linear_model import LogisticRegression

log_reg_model = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        (
            'classifier',
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]
)

log_reg_model.fit(X_train, y_train)

"""# Make Predictions"""

log_reg_pred = log_reg_model.predict(X_test)
log_reg_probability = log_reg_model.predict_proba(X_test)[:, 1]

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

results = []

results.append({
    'Model': 'Logistic Regression',
    'Accuracy': accuracy_score(y_test, log_reg_pred),
    'Precision': precision_score(y_test, log_reg_pred, zero_division=0),
    'Recall': recall_score(y_test, log_reg_pred, zero_division=0),
    'F1': f1_score(y_test, log_reg_pred, zero_division=0),
    'ROC-AUC': roc_auc_score(y_test, log_reg_probability)
})

numerical_features_check = [
    'Conversion_Rate',
    'ROI',
    'Clicks',
    'Impressions',
    'Engagement_Score',
    'Age',
    'Area_Income'
]

print(
    dataframe.groupby('Clicked_on_Ad')[
        numerical_features_check
    ].mean()
)

click_comparison = dataframe.groupby(
    'Clicked_on_Ad'
)[numerical_features_check].mean().T

click_comparison['Difference'] = (
    click_comparison[1] - click_comparison[0]
)

print(click_comparison)

"""## Random Forest Classifier"""

from sklearn.ensemble import RandomForestClassifier

rf_model = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        (
            'classifier',
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            )
        )
    ]
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_probability = rf_model.predict_proba(X_test)[:, 1]

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

print("Accuracy:", accuracy_score(y_test, rf_pred))

print("\nROC-AUC:", roc_auc_score(y_test, rf_probability))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        rf_pred,
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_pred))

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

results.append({
    'Model': 'Random Forest',
    'Accuracy': accuracy_score(y_test, rf_pred),
    'Precision': precision_score(y_test, rf_pred, zero_division=0),
    'Recall': recall_score(y_test, rf_pred, zero_division=0),
    'F1': f1_score(y_test, rf_pred, zero_division=0),
    'ROC-AUC': roc_auc_score(y_test, rf_probability)
})

"""## Naive Bayes"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(
            handle_unknown='ignore',
            sparse_output=False
        ), categorical_features)
    ]
)

from sklearn.naive_bayes import GaussianNB

nav_bayes_model = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('classifier', GaussianNB())
    ]
)

nav_bayes_model.fit(X_train, y_train)

nav_bayes_pred = nav_bayes_model.predict(X_test)
nav_bayes_probability = nav_bayes_model.predict_proba(X_test)[:, 1]

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

results.append({
    'Model': 'Naive Bayes',
    'Accuracy': accuracy_score(y_test, nav_bayes_pred),
    'Precision': precision_score(y_test, nav_bayes_pred, zero_division=0),
    'Recall': recall_score(y_test, nav_bayes_pred, zero_division=0),
    'F1': f1_score(y_test, nav_bayes_pred, zero_division=0),
    'ROC-AUC': roc_auc_score(y_test, nav_bayes_probability)
})

"""## Decision Tree"""

from sklearn.tree import DecisionTreeClassifier

dec_tree_model = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('classifier', DecisionTreeClassifier(
            random_state=42,
            max_depth=10
        ))
    ]
)

dec_tree_model.fit(X_train, y_train)

dec_tree_pred = dec_tree_model.predict(X_test)
dec_tree_probability = dec_tree_model.predict_proba(X_test)[:, 1]

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

results.append({
    'Model': 'Decision Tree',
    'Accuracy': accuracy_score(y_test, dec_tree_pred),
    'Precision': precision_score(y_test, dec_tree_pred, zero_division=0),
    'Recall': recall_score(y_test, dec_tree_pred, zero_division=0),
    'F1': f1_score(y_test, dec_tree_pred, zero_division=0),
    'ROC-AUC': roc_auc_score(y_test, dec_tree_probability)
})

results_df = pd.DataFrame(results)

results_df = results_df.round(4)

print(results_df.to_string(index=False))