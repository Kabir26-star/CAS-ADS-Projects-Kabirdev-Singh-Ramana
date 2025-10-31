#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: Divorce_Study.ipynb
Conversion Date: 2025-10-31T13:21:43.793Z
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


URL = 'https://raw.githubusercontent.com/Kabir26-star/CAS-ADS-Projects-Kabirdev-Singh-Ramana/main/CAS-ADS-Module%202/divorce_df.csv'
df = pd.read_csv(URL)
df

df_copy = df.copy()

print("Dataset copied successfully to df_copy.")
display(df_copy.head(10))

# Visualize continuous variables with histograms (excluding 'divorced') by divorced status
continuous_cols = df_copy.select_dtypes(include=np.number).drop(columns=['divorced'])
n_cols = 4  # Number of columns for subplots
n_rows = (len(continuous_cols.columns) + n_cols - 1) // n_cols  # Calculate number of rows

plt.figure(figsize=(n_cols * 5, n_rows * 4)) # Adjust figure size based on number of plots

for i, col in enumerate(continuous_cols.columns):
    plt.subplot(n_rows, n_cols, i + 1)
    sns.histplot(data=df_copy, x=col, hue='divorced', kde=True, multiple='stack') # Use hue to differentiate divorced groups

    # calculate mean values
    mean_non_div = df_copy[df_copy['divorced'] == 0][col].mean()
    mean_div = df_copy[df_copy['divorced'] == 1][col].mean()

    # Add mean Lines
    plt.axvline(mean_non_div, color='b', linestyle='--', linewidth=1.5, label='Mean Non-Divorced')
    plt.axvline(mean_div, color='r', linestyle='--', linewidth=1.5, label='Mean Divorced')


    plt.title(f'Distribution of {col} by Divorced Status')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7, color='lightgray') # Add light gray grid

plt.tight_layout()
plt.show()

# Visualize categorical and binary variables with count plots by divorced status
categorical_cols = df_copy.select_dtypes(include='object')

# Filter out columns that resulted in empty plots (based on previous execution output)
# The columns that resulted in empty Series were:
# 'cultural_background_match', 'mental_health_issues', 'infidelity_occurred',
# 'counseling_attended', 'pre_marital_cohabitation', 'domestic_violence_history'
cols_to_exclude = ['cultural_background_match', 'mental_health_issues', 'infidelity_occurred',
                   'counseling_attended', 'pre_marital_cohabitation', 'domestic_violence_history']
categorical_cols = categorical_cols.drop(columns=cols_to_exclude, errors='ignore')


n_cols = 3  # Number of columns for subplots
n_rows = (len(categorical_cols.columns) + n_cols - 1) // n_cols  # Calculate number of rows

plt.figure(figsize=(n_cols * 6, n_rows * 6)) # Adjust figure size based on number of plots

for i, col in enumerate(categorical_cols.columns):
    plt.subplot(n_rows, n_cols, i + 1)
    sns.countplot(data=df_copy, x=col, hue='divorced', order=df_copy[col].value_counts().index) # Use hue to differentiate divorced groups
    plt.title(f'Count of {col} by Divorced Status')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Divorced', labels=['Non Divorced', 'Divorced'])
    plt.grid(axis='y', linestyle='--', alpha=0.7, color='lightgray') # Add light gray grid

plt.tight_layout()
plt.show()

# Separate continuous and categorical/binary variables

# Identify binary columns (only contain 0 and 1)
binary_cols = df_copy.columns[(df_copy.nunique() == 2) & (df_copy.apply(lambda x: set(x.unique()) == {0, 1} or set(x.unique()) == {0} or set(x.unique()) == {1}).all())].tolist()

# Identify categorical columns (object dtype and not binary)
categorical_cols = df_copy.select_dtypes(include='object').columns.tolist()

# Identify continuous columns (numeric dtype and not binary)
continuous_cols = df_copy.select_dtypes(include=np.number).columns.tolist()
continuous_cols = [col for col in continuous_cols if col not in binary_cols]

print("Continuous Variables:")
display(continuous_cols)

print("\nBinary Variables:")
display(binary_cols)

print("\nCategorical Variables:")
display(categorical_cols)

# Summary for continuous variables (Mean and Standard Deviation) by Divorced Status
print("\nSummary of Continuous Variables (Mean and Standard Deviation) by Divorced Status:")
display(df_copy.groupby('divorced')[continuous_cols].agg(['mean', 'std']).T)

print("\nPercentages for Categorical and Binary Variables by Divorced Status:")
# Create a dictionary to store categorical summaries
categorical_binary_summaries = {}
for col in categorical_cols + binary_cols:
    print(f"\nColumn: {col}")
    col_summary = df_copy.groupby('divorced')[col].value_counts(normalize=True).mul(100).round(2).astype(str) + '%'
    display(col_summary)
    categorical_binary_summaries[col] = col_summary.unstack(level=0) # Unstack to get divorced status as columns

# Example of how to access a specific categorical summary DataFrame
# education_level_summary_df = categorical_binary_summaries['education_level']
# display(education_level_summary_df)

# Step 1: Separate Data by Divorced Status
df_divorced = df_copy[df_copy['divorced'] == 1]
df_nondivorced = df_copy[df_copy['divorced'] == 0]

print("Data successfully separated by divorced status.")
print(f"Divorced group shape: {df_divorced.shape}")
print(f"Non-divorced group shape: {df_nondivorced.shape}")

# Step 2: Analyze Continuous Variables

from scipy.stats import ttest_ind, probplot, shapiro, levene, mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

print("Analyzing Continuous Variables:")

# Exclude 'divorced' column from continuous variables for t-test
continuous_cols_for_test = [col for col in continuous_cols if col != 'divorced']

for col in continuous_cols_for_test:
    print(f"\nAnalyzing column: {col}")

    # Separate the data for the two groups
    group1 = df_divorced[col].dropna()  # Divorced group
    group0 = df_nondivorced[col].dropna() # Non-divorced group

    # Check for sufficient data in both groups
    if len(group1) < 2 or len(group0) < 2:
        print(f"Skipping analysis for {col} due to insufficient data in one or both groups.")
        continue

    # Test of Assumption: Normality (Shapiro-Wilk Test and QQ Plots)
    print("\nTest of Assumption: Normality (Shapiro-Wilk Test and QQ Plots)")
    print("Note: Non-parametric tests do not assume normality, but understanding the distribution is still helpful.")

    # Shapiro-Wilk Test for Normality
    # Null Hypothesis (H0): The data is drawn from a normal distribution.
    # Working Hypothesis (H1): The data is not drawn from a normal distribution.
    shapiro_group1 = shapiro(group1)
    shapiro_group0 = shapiro(group0)

    print(f"\nShapiro-Wilk Test for {col} (Divorced Group):")
    print(f"Statistic: {shapiro_group1.statistic:.4f}, P-value: {shapiro_group1.pvalue:.4f}")
    if shapiro_group1.pvalue < 0.05:
        print("Conclusion: Reject the null hypothesis. The data in the divorced group is likely not normally distributed.")
    else:
        print("Conclusion: Fail to reject the null hypothesis. The data in the divorced group may be normally distributed.")

    print(f"\nShapiro-Wilk Test for {col} (Non-Divorced Group):")
    print(f"Statistic: {shapiro_group0.statistic:.4f}, P-value: {shapiro_group0.pvalue:.4f}")
    if shapiro_group0.pvalue < 0.05:
        print("Conclusion: Reject the null hypothesis. The data in the non-divorced group is likely not normally distributed.")
    else:
        print("Conclusion: Fail to reject the null hypothesis. The data in the non-divorced group may be normally distributed.")

    # Generate QQ Plots to visually check normality assumption
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    probplot(group1, dist="norm", plot=plt)
    plt.title(f'QQ Plot for {col} (Divorced)')
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Sample Quantiles')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    probplot(group0, dist="norm", plot=plt)
    plt.title(f'QQ Plot for {col} (Non-Divorced)')
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Sample Quantiles')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Test of Assumption: Equality of Variances (Levene's Test)
    print("\nTest of Assumption: Equality of Variances (Levene's Test)")
    print("Null Hypothesis (H0): The variances of {col} are equal for divorced and non-divorced groups.")
    print("Working Hypothesis (H1): The variances of {col} are not equal for divorced and non-divorced groups.")
    levene_result = levene(group1, group0)
    print(f"Levene's Test Statistic: {levene_result.statistic:.4f}, P-value: {levene_result.pvalue:.4f}")

    if levene_result.pvalue < 0.05:
        print("Conclusion: Reject the null hypothesis. The variances are likely unequal.")
    else:
        print("Conclusion: Fail to reject the null hypothesis. The variances may be equal.")


    # Perform Mann-Whitney U Test (Non-parametric alternative to Independent Samples T-test)
    print("\nPerforming Mann-Whitney U Test:")
    print(f"Null Hypothesis (H0): The distribution of {col} is the same for divorced and non-divorced groups.")
    print(f"Working Hypothesis (H1): The distribution of {col} is different for divorced and non-divorced groups.")
    mannwhitneyu_result = mannwhitneyu(group1, group0)

    print(f"\nMann-Whitney U Test Results for {col}:")
    print(f"Statistic: {mannwhitneyu_result.statistic:.4f}")
    print(f"P-value: {mannwhitneyu_result.pvalue:.4f}")

    alpha = 0.05
    if mannwhitneyu_result.pvalue < alpha:
        print(f"Conclusion: Reject the null hypothesis. There is a significant difference in the distribution of {col} between divorced and non-divorced groups.")
    else:
        print(f"Conclusion: Fail to reject the null hypothesis. There is no significant difference in the distribution of {col} between divorced and non-divorced groups.")

    # Visualize with Histogram
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df_copy, x=col, hue='divorced', kde=True, multiple='stack')
    plt.title(f'Distribution of {col} by Status') # Updated title
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7, color='lightgray')
    plt.legend(title='Status', labels=['Non-Divorced', 'Divorced']) # Updated legend title
    plt.tight_layout()
    plt.show()

    # Visualize with Box Plot
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df_copy, x='divorced', y=col)
    plt.title(f'Distribution of {col} by Status') # Updated title
    plt.xlabel('Status') # Updated x-axis label
    plt.ylabel(col)
    plt.xticks([0, 1], ['Non-Divorced', 'Divorced'])
    plt.grid(axis='y', linestyle='--', alpha=0.7, color='lightgray')
    plt.tight_layout()
    plt.show()

# Hypothesis Testing for Categorical Variables
# Using Chi-Squared Test and Bar Plots

from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns

print("Hypothesis Testing for Categorical Variables:")

# Use the categorical_cols identified earlier
for col in categorical_cols:
    print(f"\nTesting Hypothesis for column: {col}")

    # Create a contingency table
    contingency_table = pd.crosstab(df_copy[col], df_copy['divorced'])
    print(f"Observed Contingency Table for {col} vs Divorced:")
    display(contingency_table)

    # Chi-Squared Test
    # Null Hypothesis (H0): No association between {col} and divorced status.
    # Alternative Hypothesis (H1): There is a association.
    chi2, p, dof, expected = chi2_contingency(contingency_table)

    print(f"\nChi-Squared Test Results for {col}:")
    print(f"Chi-Statistic: {chi2:.4f}")
    print(f"P-value: {p:.4f}")
    print(f"Degrees of Freedom: {dof}")

    alpha = 0.05
    if p < alpha:
        print(f"Conclusion: Reject the null hypothesis. There is a significant association between {col} and divorced status.")
    else:
        print(f"Conclusion: Fail to reject the null hypothesis. There is no significant association between {col} and divorced status.")

    # Visualize with Bar Plot
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df_copy, x=col, hue='divorced', order=df_copy[col].value_counts().index)
    plt.title(f'{col} Divorce by Status')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Divorce Status', labels=['Non-Divorced', 'Divorced']) # Update legend labels and title
    plt.grid(axis='y', linestyle='--', alpha=0.7, color='lightgray')
    plt.tight_layout()
    plt.show()

# **Machine Learning**
# **Linear Regression**
# **Marriage Duration Years**


X = df.drop(['num_children', 'education_level',      'employment_status',    'combined_income',      'religious_compatibility',      'cultural_background_match',    'communication_score',  'conflict_frequency',    'mental_health_issues', 'infidelity_occurred',  'counseling_attended',  'social_support',       'shared_hobbies_count', 'marriage_type',        'pre_marital_cohabitation',     'domestic_violence_history',    'trust_score'], axis=1)
Y = df['marriage_duration_years']

X_train, x_test, Y_train, y_test = train_test_split(X,Y,test_size=0.2, random_state = 42)

# One-hot encode the 'conflict_resolution_style' column
X_train_encoded = pd.get_dummies(X_train, columns=['conflict_resolution_style'], drop_first=True)
x_test_encoded = pd.get_dummies(x_test, columns=['conflict_resolution_style'], drop_first=True)

Scaler = StandardScaler()
X_train_scaled = Scaler.fit_transform(X_train_encoded)
x_test_scaled = Scaler.transform(x_test_encoded)

Model = LinearRegression()
Model.fit(X_train_scaled, Y_train)

# Make predictions
y_pred = Model.predict(x_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared: {r2:.4f}")

plt.figure(figsize = (10,6))
sns.regplot(x=y_test, y=y_pred, color = 'red')
plt.title("Predicted Duration of Marriage")
plt.grid(True)
plt.show()

# **Machine Learning Algorithm for Divorce**


X = df.drop(['marriage_duration_years',  'num_children', 'education_level',      'employment_status',    'combined_income',      'religious_compatibility',      'cultural_background_match',    'communication_score',  'conflict_frequency',    'mental_health_issues', 'infidelity_occurred',  'counseling_attended',  'social_support',       'shared_hobbies_count', 'marriage_type',        'pre_marital_cohabitation',     'domestic_violence_history',    'trust_score'], axis=1)
Y = df['divorced']



X_train, x_test, Y_train, y_test = train_test_split(X,Y,test_size=0.2, random_state = 42)

# One-hot encode the 'conflict_resolution_style' column
X_train_encoded = pd.get_dummies(X_train, columns=['conflict_resolution_style'], drop_first=True)
x_test_encoded = pd.get_dummies(x_test, columns=['conflict_resolution_style'], drop_first=True)

Scaler = StandardScaler()
X_train_scaled = Scaler.fit_transform(X_train_encoded)
x_test_scaled = Scaler.transform(x_test_encoded)

# One-hot encode the 'conflict_resolution_style' column
X_train_encoded = pd.get_dummies(X_train, columns=['conflict_resolution_style'], drop_first=True)
x_test_encoded = pd.get_dummies(x_test, columns=['conflict_resolution_style'], drop_first=True)

print("Data successfully one-hot encoded.")
display(X_train_encoded.head())
display(x_test_encoded.head())

model = LinearRegression()
model.fit(X_train_scaled, Y_train)

# Make predictions
y_pred = model.predict(x_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared: {r2:.4f}")

plt.figure(figsize = (10,6))
sns.regplot(x=y_test, y=y_pred, color = 'red')
plt.xlabel("Actual Divorce Status")
plt.ylabel("Predicted Divorce Status")
plt.title("Actual vs Predicted Divorce Status")
plt.grid(True)
plt.show()

# **Machine Learning Algorithm for Social Support**


X = df.drop(['num_children', 'education_level',      'employment_status',    'combined_income',      'religious_compatibility',      'cultural_background_match',    'communication_score',  'conflict_frequency',    'mental_health_issues', 'infidelity_occurred',  'counseling_attended', 'shared_hobbies_count', 'marriage_type',        'pre_marital_cohabitation',     'domestic_violence_history',    'trust_score'], axis=1)
Y = df['social_support']

X_train, x_test, Y_train, y_test = train_test_split(X,Y,test_size=0.2, random_state = 42)

Scaler = StandardScaler()
X_train_scaled = Scaler.fit_transform(X_train_encoded)
x_test_scaled = Scaler.transform(x_test_encoded)

Model = LinearRegression()
Model.fit(X_train_scaled, Y_train)

# Make predictions
y_pred = Model.predict(x_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared: {r2:.4f}")

model = LinearRegression()
model.fit(X_train_scaled, Y_train)

# Make predictions
y_pred = model.predict(x_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared: {r2:.4f}")

plt.figure(figsize = (10,6))
sns.regplot(x=y_test, y=y_pred, color = 'red')
plt.xlabel("Actual Divorce Status")
plt.ylabel("Predicted Divorce Status")
plt.title("Actual vs Predicted Divorce Status")
plt.grid(True)
plt.show()

plt.figure(figsize = (10,6))
sns.regplot(x=y_test, y=y_pred, color = 'red')
plt.title("Predicted Social Support")
plt.grid(True)
plt.show()

# **Machine Learning Algorithm for Trust Score**


X = df.drop(['marriage_duration_years',  'num_children', 'education_level',      'employment_status',    'combined_income',      'religious_compatibility',      'cultural_background_match',    'communication_score',  'conflict_frequency',    'mental_health_issues', 'infidelity_occurred',  'counseling_attended',  'social_support',       'shared_hobbies_count', 'marriage_type',        'pre_marital_cohabitation',     'domestic_violence_history', 'divorced'], axis=1)
Y = df['trust_score']

X_train, x_test, Y_train, y_test = train_test_split(X,Y,test_size=0.2, random_state = 42)

Scaler = StandardScaler()
X_train_scaled = Scaler.fit_transform(X_train_encoded)
x_test_scaled = Scaler.transform(x_test_encoded)

Model = LinearRegression()
Model.fit(X_train_scaled, Y_train)

# Make predictions
y_pred = Model.predict(x_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared: {r2:.4f}")

plt.figure(figsize = (10,6))
sns.regplot(x=y_test, y=y_pred, color = 'red')
plt.title("Predicted Trust Score")
plt.grid(True)
plt.show()

# **Machine Learning with Logistic Regression**


X = df.drop(['marriage_duration_years', 'num_children', 'cultural_background_match', 'divorced'], axis=1)
Y = df['divorced']

X_train, x_test, Y_train, y_test = train_test_split(X,Y,test_size=0.2, random_state = 42)

# Identify categorical columns to encode (excluding the target variable 'divorced')
categorical_cols_for_encoding = X_train.select_dtypes(include='object').columns.tolist()

# One-hot encode the categorical columns
X_train_encoded = pd.get_dummies(X_train, columns=categorical_cols_for_encoding, drop_first=True)
x_test_encoded = pd.get_dummies(x_test, columns=categorical_cols_for_encoding, drop_first=True)

# Scale the encoded data
Scaler = StandardScaler()
X_train_scaled = Scaler.fit_transform(X_train_encoded)
x_test_scaled = Scaler.transform(x_test_encoded)

# Create and train balanced logistic regression
log_reg_balanced = LogisticRegression(max_iter=1000, class_weight='balanced')
log_reg_balanced.fit(X_train_scaled, Y_train)

# Predict
y_pred_balanced = log_reg_balanced.predict(x_test_scaled)

# Evaluate
cm = confusion_matrix(y_test, y_pred_balanced)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Divorced", "Divorced"])
disp.plot(cmap="Blues")

#Plotting the confusion Matrix
plt.xlabel("Predicted Status")
plt.ylabel("Actual Status")
plt.title("Confusion Matrix (Balanced Logistic Regression)")
plt.show()

print("Accuracy:", accuracy_score(y_test, y_pred_balanced))
print(classification_report(y_test, y_pred_balanced, target_names=["Non-Divorced", "Divorced"]))