import urllib.request
import re
import os
import json
from pathlib import Path

CLINICAL_DIR = Path(__file__).resolve().parent.parent / "data" / "clinical"
CLINICAL_DIR.mkdir(parents=True, exist_ok=True)

def download_mendeley_file(dataset_id, output_filename):
    print(f"Fetching Mendeley dataset page for ID: {dataset_id}")
    url = f"https://data.mendeley.com/datasets/{dataset_id}/1"
    
    req = urllib.request.Request(
        url, 
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    )
    
    try:
        html = urllib.request.urlopen(req).read().decode('utf-8')
        
        # Look for the download URLs in the HTML page. 
        # Mendeley page renders files in script tags with JSON data
        matches = re.findall(r'https://data.mendeley.com/public-files/datasets/[^/]+/files/[^/]+/file_download\?download=1', html)
        if not matches:
            # Try alternate pattern
            matches = re.findall(r'href="([^"]+file_download\?download=1)"', html)
        
        if matches:
            download_url = matches[0]
            # Replace html entities if any
            download_url = download_url.replace('&amp;', '&')
            if not download_url.startswith('http'):
                download_url = 'https://data.mendeley.com' + download_url
                
            print(f"Found download URL: {download_url}")
            print(f"Downloading to {CLINICAL_DIR / output_filename}...")
            
            # Download file
            urllib.request.urlretrieve(download_url, CLINICAL_DIR / output_filename)
            print("Download complete!")
            return True
        else:
            print("Could not find download links in the page HTML.")
            return False
            
    except Exception as e:
        print(f"Error fetching/downloading: {e}")
        return False

if __name__ == "__main__":
    # Primary Mendeley clinical TB dataset ID
    download_mendeley_file("ndxdx54xxx", "mendeley_raw.xlsx")
