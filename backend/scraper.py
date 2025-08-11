import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin
from requests.packages.urllib3.exceptions import InsecureRequestWarning

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

# URL of the page to scrape
url = "https://intranet.iiit.ac.in/offices/default/display_all_files"

# Create a directory to save PDFs
os.makedirs("pdfs", exist_ok=True)

headers = {"User-Agent": "Mozilla/5.0 (compatible; PDFScraper/1.0)"}

# Fetch the page
try:
    response = requests.get(url, headers=headers, verify=False, timeout=10)
    response.raise_for_status()
except Exception as e:
    print(f"Failed to fetch page: {e}")
    exit(1)

soup = BeautifulSoup(response.content, "html.parser")

# Find all PDF links
for link in soup.find_all("a", href=True):
    href = link["href"]
    if href.lower().endswith(".pdf"):
        pdf_url = urljoin(url, href)
        filename = os.path.join("pdfs", os.path.basename(href))
        print(f"Downloading {pdf_url}...")
        try:
            pdf_response = requests.get(
                pdf_url, headers=headers, verify=False, stream=True, timeout=20
            )
            pdf_response.raise_for_status()
            with open(filename, "wb") as f:
                for chunk in pdf_response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print(f"Saved to {filename}")
        except Exception as e:
            print(f"Failed to download {pdf_url}: {e}")
