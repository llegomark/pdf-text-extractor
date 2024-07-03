import logging
from pathlib import Path
from typing import List, Tuple
import tempfile
import multiprocessing

from pdf2image import convert_from_path
import pytesseract
import cv2
import numpy as np
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
PDF_FOLDER = Path("pdf")
TEXT_FOLDER = Path("text")
IMAGES_FOLDER = Path("images")
DPI = 300
LANGUAGES = "eng+tgl"
MAX_WORKERS = multiprocessing.cpu_count()  # Use all available CPU cores


def setup_folders(pdf_folder: Path = PDF_FOLDER, text_folder: Path = TEXT_FOLDER, images_folder: Path = IMAGES_FOLDER) -> None:
    """Create necessary folders if they don't exist."""
    pdf_folder.mkdir(exist_ok=True)
    text_folder.mkdir(exist_ok=True)
    images_folder.mkdir(exist_ok=True)


def get_pdf_files(pdf_folder: Path = PDF_FOLDER) -> List[Path]:
    """Get a list of PDF files in the PDF_FOLDER."""
    return list(pdf_folder.glob("*.pdf"))


def get_image_subfolder(pdf_name: str, images_folder: Path = IMAGES_FOLDER) -> Path:
    """Get the subfolder path for saving images of a specific PDF."""
    subfolder = images_folder / pdf_name
    subfolder.mkdir(exist_ok=True)
    return subfolder


def enhance_image(image: np.ndarray) -> np.ndarray:
    """
    Apply enhancements to the image for better OCR results.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
    kernel = np.ones((1, 1), np.uint8)
    dilated = cv2.dilate(denoised, kernel, iterations=1)
    return dilated


def process_pdf(pdf_path: Path, images_folder: Path = IMAGES_FOLDER, dpi: int = DPI, languages: str = LANGUAGES) -> Tuple[str, List[str]]:
    """
    Process a single PDF file:
    1. Convert PDF to images
    2. Enhance images and save them
    3. Perform OCR on each image
    4. Combine extracted text
    """
    pdf_name = pdf_path.stem
    image_subfolder = get_image_subfolder(pdf_name, images_folder)

    with tempfile.TemporaryDirectory() as temp_dir:
        images = convert_from_path(pdf_path, dpi=dpi, output_folder=temp_dir)
        extracted_text = []

        for i, image in enumerate(images):
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            enhanced_image = enhance_image(cv_image)

            image_filename = f"page_{i+1}.png"
            cv2.imwrite(str(image_subfolder / image_filename), enhanced_image)

            text = pytesseract.image_to_string(enhanced_image, lang=languages)
            extracted_text.append(text)

            logger.debug(f"Processed and saved page {i+1} of {pdf_path.name}")

    logger.info(f"Extracted text from {pdf_path.name}: {extracted_text}")
    return pdf_name, extracted_text


def save_text(filename: str, text: List[str], text_folder: Path = TEXT_FOLDER) -> None:
    """Save extracted text to a file in the TEXT_FOLDER."""
    output_path = text_folder / f"{filename}.txt"
    with output_path.open("w", encoding="utf-8") as f:
        f.write("\n\n".join(text))
    logger.info(f"Saved text to {output_path}")


def process_pdf_wrapper(args: Tuple[Path, Path, Path]) -> None:
    """Wrapper function for parallel processing of PDFs."""
    pdf_file, images_folder, text_folder = args
    try:
        filename, extracted_text = process_pdf(pdf_file, images_folder)
        save_text(filename, extracted_text, text_folder)
        logger.info(f"Successfully processed {pdf_file.name}")
    except Exception as e:
        logger.error(f"Error processing {pdf_file.name}: {str(e)}")


def main(pdf_folder: Path = PDF_FOLDER, text_folder: Path = TEXT_FOLDER, images_folder: Path = IMAGES_FOLDER) -> None:
    """Main function to orchestrate the PDF text extraction process."""
    setup_folders(pdf_folder, text_folder, images_folder)
    pdf_files = get_pdf_files(pdf_folder)

    if not pdf_files:
        logger.warning("No PDF files found in the 'pdf' folder.")
        return

    logger.info(
        f"Found {len(pdf_files)} PDF files. Starting extraction process...")

    # Prepare arguments for multiprocessing
    process_args = [(pdf_file, images_folder, text_folder)
                    for pdf_file in pdf_files]

    # Use multiprocessing to process PDFs in parallel
    with multiprocessing.Pool(processes=MAX_WORKERS) as pool:
        list(tqdm(pool.imap_unordered(process_pdf_wrapper, process_args),
                  total=len(pdf_files), desc="Processing PDFs"))

    logger.info("PDF text extraction and image saving complete.")
    logger.info(f"Text files in output folder: {
                list(text_folder.glob('*.txt'))}")


if __name__ == "__main__":
    main()
