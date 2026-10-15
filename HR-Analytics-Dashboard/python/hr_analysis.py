import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the HR dataset
df = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Display the first 5 rows
print(df.head())

# Display dataset information
print("\nDataset Information:")
print(df.info())

# Display the number of rows and columns
print("\nDataset Shape:")
print(df.shape)

# Display column names
print("\nColumn Names:")
print(df.columns.tolist())
# Check missing values in each column
print("\nMissing Values:")
print(df.isnull().sum())

# Check total missing values
print("\nTotal Missing Values:")
print(df.isnull().sum().sum())

# Check duplicate rows
print("\nTotal Duplicate Rows:")
print(df.duplicated().sum())

# Check employee attrition counts
print("\nEmployee Attrition Counts:")
print(df["Attrition"].value_counts())

# Calculate key HR metrics

total_employees = len(df)

employees_left = (df["Attrition"] == "Yes").sum()

employees_stayed = (df["Attrition"] == "No").sum()

attrition_rate = (employees_left / total_employees) * 100

average_monthly_income = df["MonthlyIncome"].mean()

print("\n========== HR ANALYTICS SUMMARY ==========")

print("Total Employees:", total_employees)

print("Employees Who Left:", employees_left)

print("Employees Who Stayed:", employees_stayed)

print("Attrition Rate:", round(attrition_rate, 2), "%")

print("Average Monthly Income:", round(average_monthly_income, 2))

print("==========================================")
# Analyze attrition by department

department_attrition = df.groupby("Department")["Attrition"].value_counts().unstack(fill_value=0)

print("\n===== ATTRITION BY DEPARTMENT =====")
print(department_attrition)

# Calculate department-wise attrition rate
department_attrition["Attrition_Rate"] = (
    department_attrition["Yes"] /
    (department_attrition["Yes"] + department_attrition["No"])
) * 100

print("\n===== DEPARTMENT-WISE ATTRITION RATE =====")
print(department_attrition[["Attrition_Rate"]].round(2))

# Create department-wise attrition chart

import matplotlib.pyplot as plt

# Get department names and attrition rates
departments = department_attrition.index
attrition_rates = department_attrition["Attrition_Rate"]

# Create the bar chart
plt.figure(figsize=(9, 5))

plt.bar(departments, attrition_rates)

# Add chart title and labels
plt.title("Employee Attrition Rate by Department")
plt.xlabel("Department")
plt.ylabel("Attrition Rate (%)")

# Display percentage values above the bars
for i, value in enumerate(attrition_rates):
    plt.text(i, value + 0.3, f"{value:.2f}%", ha="center")

plt.tight_layout()

# Save the chart in the visualizations folder
plt.savefig("visualizations/attrition_by_department.png", dpi=300)

# Display the chart
plt.show()
# Analyze attrition by age group

# Create age groups
df["Age_Group"] = pd.cut(
    df["Age"],
    bins=[17, 25, 35, 45, 55, 65],
    labels=["18-25", "26-35", "36-45", "46-55", "56-65"]
)

# Calculate attrition rate for each age group
age_attrition = df.groupby(
    "Age_Group", observed=False
).apply(
    lambda group: (group["Attrition"] == "Yes").mean() * 100,
    include_groups=False
)

print("\nAttrition Rate by Age Group:")
print(age_attrition.round(2))

# Create the chart
plt.figure(figsize=(9, 5))

plt.bar(age_attrition.index.astype(str), age_attrition.values)

plt.title("Employee Attrition Rate by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Attrition Rate (%)")

for i, value in enumerate(age_attrition.values):
    plt.text(i, value + 0.3, f"{value:.2f}%", ha="center")

plt.tight_layout()

plt.savefig("visualizations/attrition_by_age_group.png", dpi=300)

plt.show()
# Compare average monthly income by attrition

income_comparison = df.groupby("Attrition")["MonthlyIncome"].mean()

print("\nAverage Monthly Income by Attrition:")
print(income_comparison.round(2))

# Create the chart
plt.figure(figsize=(8, 5))

plt.bar(income_comparison.index, income_comparison.values)

plt.title("Average Monthly Income: Employees Who Left vs Stayed")
plt.xlabel("Attrition")
plt.ylabel("Average Monthly Income")

for i, value in enumerate(income_comparison.values):
    plt.text(i, value + 50, f"{value:.2f}", ha="center")

plt.tight_layout()

plt.savefig("visualizations/income_by_attrition.png", dpi=300)

plt.show()
# Job Satisfaction Analysis
satisfaction = df.groupby("JobSatisfaction")["Attrition"].value_counts().unstack(fill_value=0)

# Calculate attrition rate for each satisfaction level
satisfaction["Attrition_Rate"] = (
    satisfaction["Yes"] /
    (satisfaction["Yes"] + satisfaction["No"])
) * 100

print("\nAttrition Rate by Job Satisfaction:")
print(satisfaction[["Attrition_Rate"]].round(2))

# Create chart
plt.figure(figsize=(8, 5))

plt.bar(
    satisfaction.index,
    satisfaction["Attrition_Rate"]
)

plt.title("Employee Attrition Rate by Job Satisfaction")
plt.xlabel("Job Satisfaction Level (1 = Low, 4 = High)")
plt.ylabel("Attrition Rate (%)")

for i, value in enumerate(satisfaction["Attrition_Rate"]):
    plt.text(i + 1, value + 0.3, f"{value:.2f}%", ha="center")

plt.tight_layout()
plt.savefig("visualizations/attrition_by_job_satisfaction.png", dpi=300)
plt.show()