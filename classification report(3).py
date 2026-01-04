import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv("/kaggle/input/full-dataset/Dataset .csv")

columns_to_drop = [
    'Restaurant ID', 'Restaurant Name', 'Address', 'Locality',
    'Locality Verbose', 'Switch to order menu', 'Rating color', 'Rating text'
]
df.drop(columns=columns_to_drop, inplace=True)

df.dropna(subset=['Cuisines'], inplace=True)

for col in df.columns:
    if df[col].dtype == 'object':
        df[col].fillna("Unknown", inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)

label_encoders = {}
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

X = df.drop("Cuisines", axis=1)
y = df["Cuisines"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Cuisine Classification Report:\n")
print(report)
print("Overall Accuracy:", accuracy)
