#!/usr/bin/env python
# coding: utf-8

# In[32]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("sprint1data.csv")


# In[33]:


df_encoded = df.copy()

# Encoding
age_mapping = {'16-20': 1, '20-25': 2, '25-30': 3, '30-Above': 4}
df_encoded['Age_encoded'] = df['Age'].map(age_mapping)
gender_mapping = {'Female': 0, 'Male': 1}
df_encoded['Gender_encoded'] = df['Gender'].map(gender_mapping)
occupation_dummies = pd.get_dummies(df['Occupation'], prefix='Occupation')
df_encoded = pd.concat([df_encoded, occupation_dummies], axis=1)
days_indoors_mapping = {
    'Go out Every day': 0, '1-14 days': 1, '15-30 days': 2,
    '31-60 days': 3, 'More than 2 months': 4
}
df_encoded['Days_Indoors_encoded'] = df['Days_Indoors'].map(days_indoors_mapping)
yes_no_maybe_mapping = {'No': 0, 'Maybe': 1, 'Yes': 2}
for column in ['Growing_Stress', 'Quarantine_Frustrations', 'Changes_Habits',
               'Mental_Health_History', 'Weight_Change', 'Work_Interest', 'Social_Weakness']:
    df_encoded[f'{column}_encoded'] = df[column].map(yes_no_maybe_mapping)
mood_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
df_encoded['Mood_Swings_encoded'] = df['Mood_Swings'].map(mood_mapping)
coping_mapping = {'No': 0, 'Yes': 1}
df_encoded['Coping_Struggles_encoded'] = df['Coping_Struggles'].map(coping_mapping)

# Feature Engineering
df_features = df_encoded.copy()
df_features['Stress_Index'] = df_features[['Growing_Stress_encoded', 'Quarantine_Frustrations_encoded', 'Coping_Struggles_encoded']].mean(axis=1)
df_features['Mental_Health_Vulnerability'] = df_features[['Mental_Health_History_encoded', 'Mood_Swings_encoded', 'Social_Weakness_encoded']].mean(axis=1)
df_features['Age_Isolation_Interaction'] = df_features['Age_encoded'] * df_features['Days_Indoors_encoded']
df_features['Quarantine_Impact'] = df_features['Days_Indoors_encoded'] * df_features['Quarantine_Frustrations_encoded']
df_features['High_Risk_Mental_Health'] = ((df_features['Mental_Health_History_encoded'] == 2) & (df_features['Mood_Swings_encoded'] >= 1) & (df_features['Coping_Struggles_encoded'] == 1)).astype(int)
df_features['Routine_Disruption'] = df_features['Days_Indoors_encoded'] * df_features['Changes_Habits_encoded']

df_features.head()


# In[34]:


plt.figure(figsize=(16, 12))
plt.subplot(2, 3, 1)
sns.histplot(df_features['Stress_Index'], kde=True)
plt.title('Distribution of Stress Index')

plt.subplot(2, 3, 2)
sns.histplot(df_features['Mental_Health_Vulnerability'], kde=True)
plt.title('Distribution of Mental Health Vulnerability')

plt.subplot(2, 3, 3)
sns.histplot(df_features['Age_Isolation_Interaction'], kde=True)
plt.title('Distribution of Age-Isolation Interaction')

plt.subplot(2, 3, 4)
sns.histplot(df_features['Quarantine_Impact'], kde=True)
plt.title('Distribution of Quarantine Impact')

plt.subplot(2, 3, 5)
sns.countplot(x='High_Risk_Mental_Health', data=df_features)
plt.title('Distribution of High Risk Mental Health')

plt.subplot(2, 3, 6)
sns.histplot(df_features['Routine_Disruption'], kde=True)
plt.title('Distribution of Routine Disruption')
plt.tight_layout()
plt.show()

engineered_features = ['Stress_Index', 'Mental_Health_Vulnerability', 'Age_Isolation_Interaction',
                       'Quarantine_Impact', 'High_Risk_Mental_Health', 'Routine_Disruption']
correlation_with_mood = df_features[engineered_features + ['Mood_Swings_encoded']].corr()['Mood_Swings_encoded'].sort_values(ascending=False)
print(correlation_with_mood)

