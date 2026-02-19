import os
import requests
from tqdm import tqdm
import time

# ======== CONFIGURATION ========
API_KEY = "wz57tsZ18vrNp421HZirfczUxgnk3pUH1KDuIMCNc3hb5TFon5uIOr76"
QUERY = "nature"           # Change to whatever you want
PER_PAGE = 80              # Max allowed by Pexels
MAX_PAGES = 10             # Change how many pages you want
SAVE_DIR = "pexels_images"
# ===============================

headers = {
    "Authorization": API_KEY
}

os.makedirs(SAVE_DIR, exist_ok=True)

def download_image(url, filename):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(filename, "wb") as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)

for page in range(1, MAX_PAGES + 1):
    print(f"Fetching page {page}...")

    url = f"https://api.pexels.com/v1/search?query={QUERY}&per_page={PER_PAGE}&page={page}"
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print("Error:", response.text)
        break

    data = response.json()
    photos = data.get("photos", [])

    if not photos:
        print("No more photos found.")
        break

    for photo in tqdm(photos):
        img_url = photo["src"]["original"]  # can use medium/large instead
        img_id = photo["id"]
        filename = os.path.join(SAVE_DIR, f"{img_id}.jpg")
        download_image(img_url, filename)

    time.sleep(1)  # Respect API rate limits

print("Download complete!")
