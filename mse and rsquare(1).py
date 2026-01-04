
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv("/kaggle/input/full-dataset/Dataset .csv")  


columns_to_drop = [
    'Restaurant ID', 'Restaurant Name', 'Address', 'Locality',
    'Locality Verbose', 'Switch to order menu', 'Rating color', 'Rating text'
]
df.drop(columns=columns_to_drop, inplace=True)


df.dropna(subset=['Aggregate rating'], inplace=True)


for col in df.columns:
    if df[col].dtype == 'object':
        df[col].fillna("Unknown", inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)


label_enc = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = label_enc.fit_transform(df[col])


X = df.drop("Aggregate rating", axis=1)
y = df["Aggregate rating"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R-squared (R² Score):", r2)


coefficients = pd.Series(model.coef_, index=X.columns)
important_features = coefficients.sort_values(ascending=False)
print("\nTop influential features on Aggregate Rating:\n")
print(important_features)
