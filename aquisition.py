#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
import plotly.io as pio

df = pd.read_csv("sprint1data.csv")

display(df.head())
display(df.describe().info())

