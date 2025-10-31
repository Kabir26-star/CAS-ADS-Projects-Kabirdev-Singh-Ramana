#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: notebook.ipynb
Conversion Date: 2025-10-31T13:13:36.524Z
"""

# <a href="https://colab.research.google.com/github/Kabir26-star/Module-1-Assignment/blob/main/Module_1.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


url = 'https://raw.githubusercontent.com/Kabir26-star/Module-1-Assignment/main/Healthcare.csv'
data = pd.set_option(r"display.max_rows", 200)
data = pd.read_csv(url)
data

data['Age'].plot(kind = 'hist', fill = True, histtype = 'step', color = 'blue')
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# Pie chart of the Medical Condition
data[data['Gender']== 'Male'].groupby('Medical Condition').size().plot(kind = 'pie', autopct = '%1.1f%%')
plt.title("Medical Condition Distribution for Male")
plt.show()

data[data['Gender']== 'Female'].groupby('Medical Condition').size().plot(kind = 'pie', autopct = '%1.1f%%')
plt.title("Medical Condition Distribution for Female")
plt.show()



# Calculate and display the value counts of Medical Condition grouped by Gender
medical_condition_counts = data.groupby('Gender')['Medical Condition'].value_counts()
print(medical_condition_counts)

# Define colors for each gender
gender_colors = {'Male': 'blue', 'Female': 'red'}

# Build color list
colors = [gender_colors[gender] for gender, _ in medical_condition_counts.index]

# Plotting the distribution with custom colors
ax = medical_condition_counts.plot(
    kind='bar',
    figsize=(10, 6),
    color=colors
)

plt.title("Medical Condition Distribution by Gender")
plt.xlabel("Medical Condition")
plt.ylabel("Total")
plt.legend(handles=[
    plt.Rectangle((0, 0), 1, 1, color='blue', label='Male'),
    plt.Rectangle((0, 0), 1, 1, color='Red', label='Female')
], title="Gender")
plt.show()

plt.figure(figsize = (10,5)) # Adjusting the length of the graph
for label, df_group in data.groupby('Medical Condition'):
    plt.plot(df_group['Age'], df_group['Medical Condition'], marker='o', linestyle='-', label=label) # Iterates through the Kategorie column and plots graph

# Customize the plot
plt.xlabel('Age')
plt.ylabel('Medical Condition')
plt.title('Medical Condition vs Age')
plt.legend(title='category')
plt.grid(True)

# Display the plot
plt.show()