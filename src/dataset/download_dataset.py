import os
import urllib.request

def download_dataset(url, save_dir="data", filename=None):
    os.makedirs(save_dir, exist_ok=True)
    
    if filename is None:
        filename = os.path.basename(url)
        
    file_path = os.path.join(save_dir, filename)
    
    try:
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, file_path)
        print(f"Dataset downloaded and saved to {file_path}")
    except Exception as e:
        print(f"Failed to download dataset. Error: {e}")
        return None
    
    return file_path
