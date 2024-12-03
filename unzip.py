import os
import zipfile
from tqdm import tqdm

print("Extracting dataset...")
zip_path = 'pmd_release.zip'
extract_path = ''
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    # Get list of all files in the zip
    file_list = zip_ref.namelist()
    # Show extraction progress using tqdm
    for file in tqdm(file_list):
        zip_ref.extract(file, extract_path)
print("Extraction completed!")