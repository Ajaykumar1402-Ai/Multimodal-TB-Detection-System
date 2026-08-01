import urllib.request
import re
from pathlib import Path

CLINICAL_DIR = Path(__file__).resolve().parent.parent / "data" / "clinical"
CLINICAL_DIR.mkdir(parents=True, exist_ok=True)

def download_semarang():
    url = "https://data.mendeley.com/datasets/gn4xjcdvxv/2"
    req = urllib.request.Request(
        url, 
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    )
    try:
        html = urllib.request.urlopen(req).read().decode('utf-8')
        # Search for any public-files URL
        matches = re.findall(r'https://data.mendeley.com/public-files/datasets/[^/]+/files/[^/]+/file_download[^\s"\'<>]*', html)
        if not matches:
            # Let's search with file_downloaded
            matches = re.findall(r'https://data.mendeley.com/public-files/datasets/[^/]+/files/[^/]+/file_downloaded[^\s"\'<>]*', html)
        if not matches:
            # Look for relative paths or UUIDs
            matches = re.findall(r'href="([^"]+file_download[^"]+)"', html)
            matches = [m.replace('&amp;', '&') for m in matches]
        
        if matches:
            print("Found matches:")
            for m in set(matches):
                print(f"  {m}")
            # Usually the first file or XLSX file is the main one. Let's try downloading the first one
            download_url = list(set(matches))[0]
            if not download_url.startswith('http'):
                download_url = 'https://data.mendeley.com' + download_url
                
            print(f"Downloading {download_url}...")
            # Set User-Agent headers for downloading
            dl_req = urllib.request.Request(download_url, headers={'User-Agent': 'Mozilla/5.0'})
            data = urllib.request.urlopen(dl_req).read()
            # Let's check file extension from name or use xlsx
            out_file = CLINICAL_DIR / "semarang_raw.xlsx"
            open(out_file, "wb").write(data)
            print(f"Saved to {out_file} successfully!")
            return True
        else:
            print("No file download links found in Semarang HTML.")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    download_semarang()
