#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
get_ipython().system('pip install xgboost')
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
get_ipython().system('pip install catboost')
from catboost import CatBoostRegressor

df = pd.read_csv("sprint1data.csv")


# In[23]:


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
    random_state=37
)

xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_mae = mean_absolute_error(y_test, xgb_pred)

print("XGBoost:")
print("MAE:", xgb_mae)

xgb_cv_scores = cross_val_score(xgb_model, X_encoded, y_encoded, cv=10, scoring='neg_mean_absolute_error')
print("Cross-validation MAE scores:", -xgb_cv_scores)
print("Average CV MAE:", -xgb_cv_scores.mean())


# In[24]:


#model2

cat_model = CatBoostRegressor(
    iterations=100,
    learning_rate=0.1,
    depth=3,
    loss_function='RMSE',
    random_seed=37,
    verbose=0
)

cat_model.fit(X_train, y_train)
cat_pred = cat_model.predict(X_test)

cat_mae = mean_absolute_error(y_test, cat_pred)
print("CatBoost:")
print("Mean Absolute Error:", cat_mae)

cat_cv_scores = cross_val_score(cat_model, X_encoded, y_encoded, cv=10, scoring='neg_mean_absolute_error')
print("Cross-validation MAE scores:", -cat_cv_scores)
print("Average CV MAE:", -cat_cv_scores.mean())


# In[25]:


print("XGBoost CV STD:", np.std(xgb_cv_scores))
print("CatBoost CV STD:", np.std(cat_cv_scores))

