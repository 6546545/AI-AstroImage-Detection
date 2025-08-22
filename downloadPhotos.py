import os
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm

# Configuration
DOWNLOAD_DIR = "sdss_images"
NUM_IMAGES = 5000
WIDTH = 512
HEIGHT = 512
SCALE = 0.396  # arcsec/pixel
SDSS_SQL = f"""
SELECT TOP {NUM_IMAGES} ra, dec
FROM PhotoObj
WHERE type = 3 -- 3 = galaxy, 6 = star
AND clean = 1
"""

# SDSS SkyServer URL for SQL queries
SDSS_SQL_URL = "https://skyserver.sdss.org/dr16/en/tools/search/x_sql.aspx"

def fetch_sdss_coords():
    """Fetch real RA and Dec from SDSS using SQL query."""
    response = requests.get(SDSS_SQL_URL, params={"cmd": SDSS_SQL, "format": "csv"})
    if response.status_code == 200:
        lines = response.text.strip().split("\n")
        coords = []
        for line in lines[1:]:  # skip the header
            parts = line.split(",")
            if len(parts) >= 2 and parts[0].strip() and parts[1].strip():
                try:
                    coords.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    continue
        return coords
    else:
        raise Exception(f"SDSS SQL query failed: {response.status_code}")

def cutout_url(ra, dec, scale=SCALE, width=WIDTH, height=HEIGHT):
    """Generate SDSS cutout URL."""
    return (
        "https://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg"
        f"?ra={ra}&dec={dec}&scale={scale}&width={width}&height={height}"
    )

def download_image(ra, dec, idx):
    """Download and save an image as PNG."""
    url = cutout_url(ra, dec)
    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content)).convert("RGB")
            file_path = os.path.join(DOWNLOAD_DIR, f"image_{idx+1:05d}.png")
            img.save(file_path, format="PNG")
        else:
            print(f"[{idx}] HTTP {response.status_code}")
    except Exception as e:
        print(f"[{idx}] Error: {e}")

def main():
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    print("Fetching real SDSS coordinates...")
    coords = fetch_sdss_coords()
    print(f"Fetched {len(coords)} coordinates.")

    for i, (ra, dec) in enumerate(tqdm(coords, desc="Downloading")):
        download_image(ra, dec, i)

if __name__ == "__main__":
    main()