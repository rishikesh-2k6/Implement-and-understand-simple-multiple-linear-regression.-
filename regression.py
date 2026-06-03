import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv('Titanic-Dataset.csv')

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
df['Age'] = df['Age'].fillna(df['Age'].median())

X_simple = df[['Age']]
y = df['Fare']

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_simple, y, test_size=0.2, random_state=42)

model_simple = LinearRegression()
model_simple.fit(X_train_s, y_train_s)

y_pred_s = model_simple.predict(X_test_s)

mae_s = mean_absolute_error(y_test_s, y_pred_s)
mse_s = mean_squared_error(y_test_s, y_pred_s)
r2_s = r2_score(y_test_s, y_pred_s)

print("Simple Linear Regression (Predicting Fare using Age):")
print(f"MAE: {mae_s:.4f}")
print(f"MSE: {mse_s:.4f}")
print(f"R2: {r2_s:.4f}")
print(f"Intercept: {model_simple.intercept_:.4f}")
print(f"Coefficient (Age): {model_simple.coef_[0]:.4f}")
print(f"Interpretation: For every 1-year increase in Age, the Fare is predicted to change by {model_simple.coef_[0]:.4f} units.\n")

X_multi = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']]
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_multi, y, test_size=0.2, random_state=42)

model_multi = LinearRegression()
model_multi.fit(X_train_m, y_train_m)

y_pred_m = model_multi.predict(X_test_m)

mae_m = mean_absolute_error(y_test_m, y_pred_m)
mse_m = mean_squared_error(y_test_m, y_pred_m)
r2_m = r2_score(y_test_m, y_pred_m)

print("Multiple Linear Regression (Predicting Fare using Pclass, Sex, Age, SibSp, Parch, Embarked):")
print(f"MAE: {mae_m:.4f}")
print(f"MSE: {mse_m:.4f}")
print(f"R2: {r2_m:.4f}")
print(f"Intercept: {model_multi.intercept_:.4f}")
for feature, coef in zip(X_multi.columns, model_multi.coef_):
    print(f"Coefficient ({feature}): {coef:.4f}")

plt.figure(figsize=(10, 6))
plt.scatter(X_test_s, y_test_s, color='blue', alpha=0.5, label='Actual Data')
plt.plot(X_test_s, y_pred_s, color='red', linewidth=2, label='Regression Line')
plt.title('Simple Linear Regression: Age vs Fare')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.legend()
plt.tight_layout()
plt.savefig('regression_plot.png')
