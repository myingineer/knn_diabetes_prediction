import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import contextily as ctx

# Define latitude and longitude coordinates
coords = [
    (3.71373, 7.75615),
    (3.71581, 7.75773),
    (3.71848, 7.75944),
    (3.72083, 7.76114),
    (3.72345, 7.76285),
    (3.72508, 7.76156),
    (3.71432, 7.75568)
]

# Create GeoDataFrame
gdf = gpd.GeoDataFrame(geometry=[Point(lon, lat) for lon, lat in coords], crs="EPSG:4326")

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
gdf.plot(ax=ax, color='red', marker='o', label="Locations")

# Add a basemap for better visualization
ctx.add_basemap(ax, crs=gdf.crs, source=ctx.providers.OpenStreetMap.Mapnik)

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Mapped Locations in Oyo State, Nigeria")
plt.legend()
plt.grid(True)
plt.show()
