# download_data.py
# Run at build time on Render to fetch large data files from Google Drive.

import os
import gdown

FILES = {
    "outputs/graph_reconstruction/traffic_state_scored.geojson": "1qKfDJTIWCBq9l10MHZ-4OWdX5YRdTtNG",
    "outputs/graph_reconstruction/gurugram_traffic_arrays.pkl":  "19ClZym3IxIfwiTqYIwqQXTi71OHqmxZp",
    "outputs/networks/full.net.xml":                             "1tbfmvTyfF01kDCpAOJ5YiiwS8SarLWpG",
    "outputs/networks/movements.pkl":                            "10qKVzbK-gwHooTtV2TTNaaqhZ1Y5iYIR",
}

for dest_path, file_id in FILES.items():
    if os.path.exists(dest_path):
        print(f"  Already exists, skipping: {dest_path}")
        continue
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"  Downloading → {dest_path}")
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, dest_path, quiet=False)

print("All data files ready.")
