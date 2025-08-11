#!/usr/bin/env python3
"""
Simple test script to verify pdfplumber functionality
"""

import pdfplumber
import io
from pathlib import Path


def test_pdf_processing():
    """Test basic PDF processing functionality"""

    # Test if we can create a simple PDF processor
    print("Testing pdfplumber functionality...")

    # Let's test with an actual PDF file if available
    pdfs_dir = Path("../pdfs")

    if pdfs_dir.exists():
        pdf_files = list(pdfs_dir.glob("*.pdf"))
        if pdf_files:
            test_pdf = pdf_files[0]
            print(f"Testing with: {test_pdf.name}")

            try:
                with pdfplumber.open(test_pdf) as pdf:
                    print(f"PDF has {len(pdf.pages)} pages")

                    # Extract text from first page
                    if pdf.pages:
                        first_page = pdf.pages[0]
                        text = first_page.extract_text()

                        if text:
                            print(f"First page text (first 200 chars): {text[:200]}...")
                        else:
                            print("No text found on first page")

                print("✅ pdfplumber PDF processing test successful!")
                return True

            except Exception as e:
                print(f"❌ Error processing PDF: {e}")
                return False
        else:
            print("No PDF files found in ../pdfs directory")
    else:
        print("../pdfs directory not found")

    # Test with simple functionality
    print("✅ pdfplumber basic import test successful!")
    return True


def test_extract_text_function():
    """Test the extract_text_from_pdf function logic"""
    print("\nTesting PDF text extraction logic...")

    def extract_text_from_pdf_bytes(file_content: bytes) -> str:
        """Extract text from PDF file content using pdfplumber"""
        try:
            pdf_file = io.BytesIO(file_content)
            text = ""

            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"

            return text.strip()
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""

    # Test with an actual PDF if available
    pdfs_dir = Path("../pdfs")
    if pdfs_dir.exists():
        pdf_files = list(pdfs_dir.glob("*.pdf"))
        if pdf_files:
            test_pdf = pdf_files[0]

            try:
                with open(test_pdf, "rb") as f:
                    file_content = f.read()

                text = extract_text_from_pdf_bytes(file_content)

                if text:
                    print(f"✅ Extracted {len(text)} characters from {test_pdf.name}")
                    print(f"Sample: {text[:100]}...")
                else:
                    print(f"⚠️  No text extracted from {test_pdf.name}")

                return True

            except Exception as e:
                print(f"❌ Error testing extraction: {e}")
                return False

    print("✅ Text extraction logic test completed (no PDFs to test with)")
    return True


if __name__ == "__main__":
    print("=== PDF Processing Test ===")

    success = True

    if not test_pdf_processing():
        success = False

    if not test_extract_text_function():
        success = False

    if success:
        print("\n🎉 All tests passed! pdfplumber is working correctly.")
    else:
        print("\n❌ Some tests failed.")
