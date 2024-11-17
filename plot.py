import pandas as pd
import folium
import folium.plugins

# Read data into DataFrame
df = pd.read_csv("result_test.csv")

# Drop rows with missing latitude or longitude
#df = df.dropna(subset=['Location.GIS.Latitude', 'Location.GIS.Longitude'])

# Initialize a Folium map centered around the mean coordinates
center_lat = df["Location.GIS.Latitude"].mean()
center_lon = df["Location.GIS.Longitude"].mean()
m = folium.Map(location=[center_lat, center_lon], zoom_start=8)
marker_cluster = folium.plugins.MarkerCluster().add_to(m)

# Add markers to the map using zip for better performance
latitudes = df["Location.GIS.Latitude"]
longitudes = df["Location.GIS.Longitude"]
ids = df["Listing.ListingId"]
prices = df["Listing.Price.ClosePrice"]

for lat, lon, listing_id, price in zip(latitudes, longitudes, ids, prices):
    tooltip_html = f"""
    <div style="font-size:16px; font-weight:bold;">
        ID: {listing_id}<br>
        Price: ${price:,.2f}
    </div>
    """
    folium.CircleMarker(
        location=[lat, lon],
        radius=price / 10000,  # Scale radius based on price
        color='red',
        fill=True,
        fill_opacity=0.6,
        tooltip=folium.Tooltip(tooltip_html)
    ).add_to(marker_cluster)


m.save("clustered.html")
