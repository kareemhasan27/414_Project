#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

from catboost import CatBoostRegressor
df = pd.read_csv("sprint1data.csv")


# In[9]:


# Occupation
plt.subplot(3, 3, 3)
sns.countplot(x='Occupation', data=df)
plt.title('Distribution of Occupation')
plt.xticks(rotation=45)

# Age and Mood Swings
plt.figure(figsize=(8, 4))
sns.boxplot(x='Mood_Swings', y='Age', data=df)
plt.title('Age vs Mood Swings')
plt.show()

plt.subplot()
sns.countplot(x='Age', hue='Mood_Swings', data=df)
plt.title('Age vs Mood Swings')
plt.xticks(rotation=45)
plt.tight_layout()

sns.countplot(x='Coping_Struggles', hue='Gender', data=df)
plt.title('Coping Struggles by Gender')
plt.show()

sns.barplot(data=df, x="Growing_Stress", y="Mood_Swings")
plt.title("Growing Stress vs Mood Swings")
plt.show()

plt.subplot(1, 1, 1)
sns.countplot(x='Growing_Stress', hue='Mood_Swings', data=df)
plt.title('Relationship between Growing Stress and Mood Swings')
plt.xticks(rotation=45)
plt.show()

plt.subplot(1, 1, 1)
sns.countplot(x='Mental_Health_History', hue='Mood_Swings', data=df)
plt.title('Mental Health History vs Mood Swings')


# In[11]:


X = df[["Growing_Stress", "Weight_Change", "Social_Weakness"]]
y = df["Mood_Swings"]

encoder = LabelEncoder()
X_encoded = X.copy()
for col in X_encoded.columns:
    X_encoded[col] = encoder.fit_transform(X_encoded[col])
y_encoded = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)

xgb_model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    random_state=42
)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_mae = mean_absolute_error(y_test, xgb_pred)
xgb_cv_scores = cross_val_score(xgb_model, X_encoded, y_encoded, cv=10, scoring='neg_mean_absolute_error')


# In[12]:


cat_model = CatBoostRegressor(
    iterations=100,
    learning_rate=0.1,
    depth=3,
    loss_function='RMSE',
    random_seed=42,
    verbose=0
)
cat_model.fit(X_train, y_train)
cat_pred = cat_model.predict(X_test)
cat_mae = mean_absolute_error(y_test, cat_pred)
cat_cv_scores = cross_val_score(cat_model, X_encoded, y_encoded, cv=10, scoring='neg_mean_absolute_error')


# In[14]:


# --- compare viz ---
# XGBoost Feature Importance
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
xgb_importances = xgb_model.feature_importances_
feat_names = X_encoded.columns
indices = np.argsort(xgb_importances)
plt.barh(range(len(indices)), xgb_importances[indices], align='center', color='blue')
plt.yticks(range(len(indices)), [feat_names[i] for i in indices])
plt.xlabel('Feature Importance')
plt.title('XGBoost Feature Importance')

# CatBoost Feature Importance
plt.subplot(1, 2, 2)
cat_importances = cat_model.get_feature_importance()
indices = np.argsort(cat_importances)
plt.barh(range(len(indices)), cat_importances[indices], align='center', color='green')
plt.yticks(range(len(indices)), [feat_names[i] for i in indices])
plt.xlabel('Feature Importance')
plt.title('CatBoost Feature Importance')
plt.tight_layout()
plt.show()

# Model MAE comparison
models = ['Decision Tree (Sprint 1)', 'XGBoost', 'CatBoost']
mae_values = [1.0, xgb_mae, cat_mae]
cv_mae_values = [None, -xgb_cv_scores.mean(), -cat_cv_scores.mean()]
bar_width = 0.35
index = np.arange(len(models))
plt.bar(index, mae_values, bar_width, label='Single Split MAE', color='royalblue')

cv_index = index[1:]
cv_values = cv_mae_values[1:]
plt.bar(cv_index + bar_width, cv_values, bar_width, label='10-fold CV MAE', color='lightcoral')

plt.xlabel('Models')
plt.ylabel('Mean Absolute Error (MAE)')
plt.title('Model Performance Comparison')
plt.xticks(index + bar_width/2, models)
plt.ylim(0, 1.1)
plt.legend()
plt.tight_layout()
plt.show()

