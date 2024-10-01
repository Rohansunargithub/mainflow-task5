# Heart Disease Analysis Project

# Importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('heart_disease_dataset.csv')

# Data Cleaning

missing_values = df.isnull().sum()
print("Missing Values:\n", missing_values)
print("\nDescriptive Statistics:\n", df.describe())

# for outliers using IQR
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df_cleaned = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
print("\nNumber of rows after removing outliers:", len(df_cleaned))

# Exploratory Data Analysis (EDA)

correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Age distribution
sns.histplot(df['age'], kde=True)
plt.title('Age Distribution')
plt.show()

# Relationship between cholesterol and heart disease
sns.boxplot(x='heart_disease', y='cholesterol', data=df)
plt.title('Cholesterol Levels and Heart Disease')
plt.show()

# Blood pressure vs heart disease
sns.boxplot(x='heart_disease', y='resting_blood_pressure', data=df)
plt.title('Blood Pressure and Heart Disease')
plt.show()

# Answering Questions

# Question 1: Correlation between age and heart disease
age_corr = df[['age', 'heart_disease']].corr()
print("\nCorrelation between age and heart disease:\n", age_corr)

# Question 2: Cholesterol and heart disease
sns.boxplot(x='heart_disease', y='cholesterol', data=df)
plt.title('Cholesterol and Heart Disease')
plt.show()

# Question 3: Blood pressure and heart disease
sns.boxplot(x='heart_disease', y='resting_blood_pressure', data=df)
plt.title('Blood Pressure and Heart Disease')
plt.show()

# Question 4: Are males more likely to have heart disease than females?
gender_heart_disease = df.groupby('sex')['heart_disease'].mean()
print("\nHeart disease rates by gender:\n", gender_heart_disease)

# Question 5: Is there a specific age range where heart disease is most prevalent?
age_groups = pd.cut(df['age'], bins=[20, 40, 60, 80])
age_heart_disease = df.groupby(age_groups)['heart_disease'].mean()
print("\nHeart disease rates by age group:\n", age_heart_disease)

# Question 6: Does smoking impact heart disease occurrence?
smoking_heart_disease = df.groupby('smoking')['heart_disease'].mean()
print("\nHeart disease rates by smoking status:\n", smoking_heart_disease)

# Question 7: Can we predict heart disease occurrence using logistic regression?

# Prepare data for prediction
X = df[['age', 'sex', 'cholesterol', 'resting_blood_pressure', 'max_heart_rate_achieved', 'smoking', 'diabetes']]
y = df['heart_disease']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)

# Make predictions
y_pred = logreg.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nLogistic Regression Accuracy:", accuracy)

# Data Visualization

# Age vs Cholesterol colored by heart disease
sns.scatterplot(x='age', y='cholesterol', hue='heart_disease', data=df)
plt.title('Age vs Cholesterol (Colored by Heart Disease)')
plt.show()

# Final Summary
print("\nSummary of findings:")
print("1. Weak correlation between age and heart disease (-0.0287)")
print("2. Males have slightly higher heart disease rates (53.7%) compared to females (44.6%)")
print("3. Heart disease prevalence is similar across age groups")
print("4. Smokers have slightly higher heart disease rates than non-smokers")
print("5. Logistic Regression model accuracy is 50%, suggesting room for improvement in prediction models")
