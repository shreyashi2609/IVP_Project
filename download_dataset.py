import ee
import geemap
import os

ee.Initialize(project='ivpproject-507812')

# Focused coordinate box in Rondônia, Brazil
roi = ee.Geometry.BBox(-63.5, -10.5, -63.0, -10.0)
output_dir = "./amazon_30years_clean"
os.makedirs(output_dir, exist_ok=True)

for year in range(1994, 2025):
    # Select satellite collection and band order
    if year <= 2011:
        collection_id = "LANDSAT/LT05/C02/T1_TOA"
        bands = ['B3', 'B2', 'B1']  # Landsat 5 RGB
    elif year == 2012:
        collection_id = "LANDSAT/LE07/C02/T1_TOA"
        bands = ['B3', 'B2', 'B1']  # Landsat 7 RGB
    else:
        collection_id = "LANDSAT/LC08/C02/T1_TOA"
        bands = ['B4', 'B3', 'B2']  # Landsat 8/9 RGB

    # 1. Filter across the Amazon dry season (June to October)
    collection = (ee.ImageCollection(collection_id)
                  .filterBounds(roi)
                  .filterDate(f"{year}-06-01", f"{year}-10-31")
                  .filter(ee.Filter.lt('CLOUD_COVER', 40)))

    # 2. .median() stitches scenes together and removes clouds/gaps
    median_composite = collection.select(bands).median()

    # 3. Standardize 8-bit RGB visualization for OpenCV
    rgb_image = median_composite.visualize(min=0.03, max=0.25)

    out_file = os.path.join(output_dir, f"rondonia_{year}.tif")
    
    try:
        # scale=60 speeds up download while maintaining crisp resolution for Otsu
        geemap.ee_export_image(rgb_image, filename=out_file, scale=60, region=roi)
        print(f"[{year}] Successfully downloaded clean composite.")
    except Exception as e:
        print(f"[{year}] Error: {e}")