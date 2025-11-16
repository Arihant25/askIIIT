download_pdfs.py - README

What
----
Small helper to download files (default PDFs) linked from a webpage.

Requirements
------------
- Python 3.8+
- pip packages: requests, beautifulsoup4, tqdm

Install
-------
Use the project's virtualenv (recommended):

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# or install ad-hoc
pip install requests beautifulsoup4 tqdm
```

How to use
----------

Default (downloads .pdf):

```bash
python scripts/download_pdfs.py https://www.example.com/offices
```

Specify file types, concurrency and output folder:

```bash
python scripts/download_pdfs.py https://www.example.com/offices -e .pdf,.docx -c 10 -o myfiles
```

Disable progress bar:

```bash
python scripts/download_pdfs.py https://www.example.com/offices --no-progress
```

Notes
-----
- Script normalizes relative links using the page base URL.
- Uses a retry HTTP adapter for resilience.
- If Content-Disposition has a filename, script will prefer it when saving.
- Avoid scraping pages where you don't have permission.
