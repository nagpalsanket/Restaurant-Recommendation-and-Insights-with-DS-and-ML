import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity

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

df['Original_Cuisines'] = df['Cuisines']

le_dict = {}
for col in df.columns:
    if df[col].dtype == 'object' and col != 'Cuisines':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

cuisine_le = LabelEncoder()
df['Cuisines'] = cuisine_le.fit_transform(df['Cuisines'])

X = df.drop(["Aggregate rating", "Original_Cuisines"], axis=1)
y = df["Aggregate rating"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

y_pred = reg_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Regression Results:")
print("Mean Squared Error (MSE):", mse)
print("R-squared (R² Score):", r2)

coefficients = pd.Series(reg_model.coef_, index=X.columns)
print("Top 5 Influential Features on Rating:")
print(coefficients.sort_values(ascending=False).head(5))

recommendation_df = df[['Cuisines', 'Price range', 'Aggregate rating']].copy()

def recommend_restaurants(user_cuisine: str, user_price: int, top_n=5):
    try:
        cuisine_encoded = cuisine_le.transform([user_cuisine])[0]
    except:
        print(f"{user_cuisine} not found in cuisine list. Try one of these:")
        print(list(cuisine_le.classes_))
        return pd.DataFrame()
    
    user_vector = np.array([[cuisine_encoded, user_price, df['Aggregate rating'].mean()]])
    similarities = cosine_similarity(user_vector, recommendation_df)[0]
    top_indices = similarities.argsort()[-top_n:][::-1]

    return df.iloc[top_indices][['Original_Cuisines', 'Price range', 'Aggregate rating']].assign(Similarity=similarities[top_indices])

print("Recommended Restaurants for 'North Indian' and Price 2:")
print(recommend_restaurants("North Indian", 2))
