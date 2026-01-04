import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium

df = pd.read_csv("/kaggle/input/full-dataset/Dataset .csv")

df.dropna(subset=['Latitude', 'Longitude', 'Locality'], inplace=True)

df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
df.dropna(subset=['Latitude', 'Longitude'], inplace=True)

avg_ratings_by_locality = df.groupby('Locality')['Aggregate rating'].mean().sort_values(ascending=False)
restaurant_count_by_locality = df['Locality'].value_counts()
avg_price_by_locality = df.groupby('Locality')['Price range'].mean().sort_values(ascending=False)

map_center = [df['Latitude'].mean(), df['Longitude'].mean()]
restaurant_map = folium.Map(location=map_center, zoom_start=11)

for _, row in df.iterrows():
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=2,
        color='red',
        fill=True,
        fill_opacity=0.4
    ).add_to(restaurant_map)

plt.figure(figsize=(12, 6))
top_localities = restaurant_count_by_locality.head(15)
sns.barplot(x=top_localities.values, y=top_localities.index, palette="viridis")
plt.xlabel("Number of Restaurants")
plt.ylabel("Locality")
plt.title("Top 15 Localities with Most Restaurants")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
top_avg_ratings = avg_ratings_by_locality.head(15)
sns.barplot(x=top_avg_ratings.values, y=top_avg_ratings.index, palette="mako")
plt.xlabel("Average Rating")
plt.ylabel("Locality")
plt.title("Top 15 Localities with Highest Average Ratings")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
top_avg_prices = avg_price_by_locality.head(15)
sns.barplot(x=top_avg_prices.values, y=top_avg_prices.index, palette="rocket")
plt.xlabel("Average Price Range")
plt.ylabel("Locality")
plt.title("Top 15 Expensive Localities (Avg. Price Range)")
plt.tight_layout()
plt.show()

restaurant_map
