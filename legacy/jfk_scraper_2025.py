import os
import requests
import zipfile
from bs4 import BeautifulSoup
from tqdm import tqdm
import time
from urllib.parse import urljoin

# Base URL for JFK document release
BASE_URL = "https://www.archives.gov/research/jfk/release-2025"
PAGE_URL_TEMPLATE = BASE_URL + "?page={}"  # Pagination template

# Directory for saving PDFs
SAVE_DIR = "JFK_Documents"
ZIP_FILE = "JFK_Documents.zip"

# Create save directory if it doesn't exist
os.makedirs(SAVE_DIR, exist_ok=True)

def download_pdf_with_retries(pdf_url, pdf_name, max_retries=5):
    """Download a PDF file with retry attempts."""
    retries = 0
    while retries < max_retries:
        try:
            response = requests.get(pdf_url, stream=True, timeout=10)
            if response.status_code == 200:
                pdf_path = os.path.join(SAVE_DIR, pdf_name)
                with open(pdf_path, "wb") as pdf_file:
                    for chunk in response.iter_content(1024):
                        pdf_file.write(chunk)
                print(f" Successfully downloaded: {pdf_name}")
                return
            else:
                print(f"Failed to download {pdf_name}, status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f" Error downloading {pdf_name}: {e}")

        retries += 1
        print(f" Retrying {pdf_name}... ({retries}/{max_retries})")
        time.sleep(3)  # Wait before retrying

    print(f" Failed to download {pdf_name} after {max_retries} attempts.")

def get_pdf_links_from_page(page):
    """Scrape direct PDF links from a single page."""
    url = PAGE_URL_TEMPLATE.format(page)
    print(f"🔎 Scraping page: {url}")

    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            print(f" Failed to fetch {url} (Status: {response.status_code})")
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        pdf_links = []

        # Find all <a> links that point to PDFs
        for link in soup.find_all("a", href=True):
            href = link["href"]
            if href.lower().endswith(".pdf"):  # Ensure it's a PDF file
                full_url = urljoin(BASE_URL, href)  # Convert relative URL to absolute
                pdf_links.append(full_url)
        
        return pdf_links

    except requests.exceptions.RequestException as e:
        print(f" Error fetching page {page}: {e}")
        return []

def download_pdfs_from_page(pdf_links):
    """Downloads PDFs from a list of links with retries."""
    for pdf_url in tqdm(pdf_links, desc="📥 Downloading PDFs"):
        pdf_name = os.path.basename(pdf_url)
        pdf_path = os.path.join(SAVE_DIR, pdf_name)

        if os.path.exists(pdf_path):
            print(f" Skipping {pdf_name}, already downloaded.")
            continue

        download_pdf_with_retries(pdf_url, pdf_name)

def create_zip():
    """Zips all downloaded PDFs into a single file."""
    with zipfile.ZipFile(ZIP_FILE, "w") as zipf:
        for file in os.listdir(SAVE_DIR):
            zipf.write(os.path.join(SAVE_DIR, file), arcname=file)
    print(f" Zipped all PDFs into {ZIP_FILE}")

# Run the script
page = 0
while True:
    pdf_links = get_pdf_links_from_page(page)
    
    if not pdf_links:  
        print("No more PDFs found. Stopping.")
        break  # Stop when there are no more PDFs

    download_pdfs_from_page(pdf_links)
    page += 1  # Move to the next page

# After all pages are processed, create the zip file
create_zip()
print("🎉 All PDFs downloaded and zipped successfully!")
