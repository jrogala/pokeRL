import zipfile
import argparse
import urllib.request
import os
import uuid
from pathlib import Path

current_folder = Path(os.path.dirname(os.path.realpath(__file__)), "../..")

def constant_hash(url: str):
    return uuid.uuid5(uuid.NAMESPACE_URL, url)

def download_rom(url: str):
    url_hash = constant_hash(url)
    zip_file_path = Path(current_folder, f"rom/{url_hash}.zip")

    # Download the file from the URL
    with urllib.request.urlopen(url) as romzip:
        file = romzip.read()
        with zip_file_path.open("wb") as f:
            f.write(file)

def extract_rom(url: str):
    url_hash = constant_hash(url)
    zip_file_path = Path(current_folder, f"rom/{url_hash}.zip")

    # Specify the destination directory to extract the files
    destination_directory = "rom"

    # Open the ZIP file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        # Extract all the files to the destination directory
        zip_ref.extractall(destination_directory)

def remove_rom(url: str):
    url_hash = constant_hash(url)
    # Specify the path to the ZIP file
    zip_file_path = Path(current_folder, f"rom/{url_hash}.zip")

    # Remove the ZIP file
    os.remove(zip_file_path)

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Download and extract the ROM")

    # Add the arguments
    parser.add_argument("-d", "--download", action="store_true", help="Download the ROM")
    parser.add_argument("-e", "--extract", action="store_true", help="Extract the ROM")
    parser.add_argument("-u", "--url", type=str, help="Specify the URL of the ROM", default="")
    # Parse the arguments
    args = parser.parse_args()

    url = args.url if args.url else "https://archive.org/download/No-Intro-Collection_2016-01-03_Fixed/Nintendo%20-%20Game%20Boy.zip/Pokemon%20-%20Blue%20Version%20%28USA%2C%20Europe%29%20%28SGB%20Enhanced%29.zip"
    # Download the ROM
    try:
        if args.download:
            download_rom(url)

        # Extract the ROM
        if args.extract:
            extract_rom(url)
    finally:
        if args.download and args.extract:
            remove_rom(url)

if __name__ == "__main__":
    main()
