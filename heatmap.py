import folium
from folium.plugins import HeatMap
import pandas as pd

df = pd.read_csv("result_test.csv")

# Create a base map
center_lat = df["Location.GIS.Latitude"].mean()
center_lon = df["Location.GIS.Longitude"].mean()
m = folium.Map(location=[center_lat, center_lon], zoom_start=10)

# Prepare the data for the HeatMap
heat_data = [[row['Location.GIS.Latitude'], row['Location.GIS.Longitude'], row['Listing.Price.ClosePrice']] for _, row in df.iterrows()]

# Add the HeatMap
HeatMap(heat_data, min_opacity=0.7, radius=10).add_to(m)

# Save or display the map
m.save("heatmap.html")
