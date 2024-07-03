# PDF Text Extractor

PDF Text Extractor is a Python-based tool that extracts text from PDF files using OCR (Optical Character Recognition). It supports both English and Tagalog languages, processes multiple PDFs in parallel, and saves both the extracted text and enhanced images.

## Features
- Extract text from PDF files using Tesseract OCR
- Support for English and Tagalog languages
- Parallel processing of multiple PDFs for improved performance
- Image enhancement for better OCR results
- Save extracted text as .txt files
- Save enhanced images as .png files

## Requirements
- Python 3.12+
- pdf2image
- pytesseract
- opencv-python (cv2)
- numpy
- tqdm
- reportlab (for testing)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/llegomark/pdf-text-extractor.git
   cd pdf-text-extractor
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Install Tesseract OCR on your system:
   - For Ubuntu:
     ```
     sudo apt-get install tesseract-ocr
     ```
   - For other operating systems, please refer to the [Tesseract documentation](https://github.com/tesseract-ocr/tesseract).

## Usage

1. Place your PDF files in the `pdf` folder.

2. Run the script:
   ```
   python pdf_extractor.py
   ```

3. The extracted text will be saved in the `text` folder, and the enhanced images will be saved in the `images` folder.

## Configuration

You can modify the following constants in `pdf_extractor.py` to customize the behavior:

- `PDF_FOLDER`: Path to the folder containing input PDF files (default: "pdf")
- `TEXT_FOLDER`: Path to the folder where extracted text will be saved (default: "text")
- `IMAGES_FOLDER`: Path to the folder where enhanced images will be saved (default: "images")
- `DPI`: DPI for PDF to image conversion (default: 300)
- `LANGUAGES`: Languages for OCR (default: "eng+tgl" for English and Tagalog)
- `MAX_WORKERS`: Number of parallel processes to use (default: number of CPU cores)

## Testing

To run the tests:

```
pytest test_pdf_extractor.py -v
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.