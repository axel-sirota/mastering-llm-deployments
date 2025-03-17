#!/bin/bash
# download_locals.sh
# This script downloads a ZIP archive from a specified URL using wget,
# saves it as locals.zip, and unzips it into local_models and local_datasets folders.

ZIP_URL="https://your-download-url/locals.zip"  # Replace with your actual URL
echo "Downloading ZIP archive from ${ZIP_URL}..."
wget -O locals.zip "${ZIP_URL}"
echo "Download complete. Unzipping..."
unzip locals.zip -d .
echo "Unzipping complete. Check that local_models and local_datasets folders exist."
