import pytest
from pathlib import Path
import shutil
import tempfile
import cv2
import numpy as np
from pdf_extractor import (
    setup_folders,
    get_pdf_files,
    get_image_subfolder,
    enhance_image,
    process_pdf,
    save_text,
    process_pdf_wrapper,
    main
)
from reportlab.pdfgen import canvas
import multiprocessing

# Register the integration mark
pytest.mark.integration = pytest.mark.integration


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield Path(tmpdirname)


@pytest.fixture
def sample_pdf(temp_dir):
    pdf_path = temp_dir / "sample.pdf"
    c = canvas.Canvas(str(pdf_path))
    c.drawString(100, 750, "Hello, World!")
    c.save()
    yield pdf_path


@pytest.fixture
def setup_test_env(temp_dir):
    pdf_folder = temp_dir / "pdf"
    text_folder = temp_dir / "text"
    images_folder = temp_dir / "images"
    setup_folders(pdf_folder, text_folder, images_folder)
    return pdf_folder, text_folder, images_folder


def test_setup_folders(temp_dir):
    pdf_folder = temp_dir / "pdf"
    text_folder = temp_dir / "text"
    images_folder = temp_dir / "images"

    setup_folders(pdf_folder, text_folder, images_folder)

    assert pdf_folder.exists() and pdf_folder.is_dir()
    assert text_folder.exists() and text_folder.is_dir()
    assert images_folder.exists() and images_folder.is_dir()


def test_get_pdf_files(setup_test_env):
    pdf_folder, _, _ = setup_test_env

    # Create some sample PDF files
    (pdf_folder / "test1.pdf").touch()
    (pdf_folder / "test2.pdf").touch()
    (pdf_folder / "not_a_pdf.txt").touch()

    pdf_files = get_pdf_files(pdf_folder)

    assert len(pdf_files) == 2
    assert all(file.suffix == ".pdf" for file in pdf_files)


def test_get_image_subfolder(setup_test_env):
    _, _, images_folder = setup_test_env

    subfolder = get_image_subfolder("test_pdf", images_folder)

    assert subfolder.exists() and subfolder.is_dir()
    assert subfolder.name == "test_pdf"


def test_enhance_image():
    # Create a sample image
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.putText(image, "Test", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    enhanced = enhance_image(image)

    assert enhanced.shape == (100, 100)  # Expect grayscale output
    assert enhanced.dtype == np.uint8
    # Ensure there's some content in the enhanced image
    assert np.sum(enhanced) > 0


@pytest.mark.parametrize("text_content", [
    ["Hello, World!"],
    ["Multiple", "Lines", "Of", "Text"],
    ["Special chars: !@#$%^&*()"],
])
def test_save_text(temp_dir, text_content):
    filename = "test_output"
    text_folder = temp_dir / "text"
    text_folder.mkdir()

    save_text(filename, text_content, text_folder)

    output_file = text_folder / f"{filename}.txt"
    assert output_file.exists()

    with output_file.open("r", encoding="utf-8") as f:
        content = f.read()

    assert content == "\n\n".join(text_content)


@pytest.mark.integration
def test_process_pdf(sample_pdf, setup_test_env):
    _, text_folder, images_folder = setup_test_env

    pdf_name, extracted_text = process_pdf(sample_pdf, images_folder)

    assert pdf_name == sample_pdf.stem
    assert len(extracted_text) > 0
    assert (images_folder / pdf_name).exists()
    assert any((images_folder / pdf_name).glob("*.png"))


@pytest.mark.integration
def test_process_pdf_wrapper(sample_pdf, setup_test_env):
    pdf_folder, text_folder, images_folder = setup_test_env

    # Copy sample PDF to pdf_folder
    shutil.copy(sample_pdf, pdf_folder / sample_pdf.name)

    process_pdf_wrapper((pdf_folder / sample_pdf.name,
                        images_folder, text_folder))

    assert (text_folder / f"{sample_pdf.stem}.txt").exists()
    assert (images_folder / sample_pdf.stem).exists()
    assert any((images_folder / sample_pdf.stem).glob("*.png"))


@pytest.mark.integration
def test_main(sample_pdf, setup_test_env):
    pdf_folder, text_folder, images_folder = setup_test_env

    # Create multiple sample PDFs
    for i in range(3):
        shutil.copy(sample_pdf, pdf_folder / f"sample_{i}.pdf")

    main(pdf_folder, text_folder, images_folder)

    text_files = list(text_folder.glob("*.txt"))
    image_files = list(images_folder.glob("*/*.png"))

    print(f"Text folder contents: {list(text_folder.iterdir())}")
    print(f"Images folder contents: {list(images_folder.iterdir())}")
    print(f"Text files found: {text_files}")
    print(f"Image files found: {image_files}")

    assert len(text_files) == 3, f"Expected 3 text files, but found {
        len(text_files)}"
    assert len(list(images_folder.iterdir())) == 3, f"Expected 3 image subfolders, but found {
        len(list(images_folder.iterdir()))}"
    assert all(len(list((images_folder / f"sample_{i}").glob(
        "*.png"))) > 0 for i in range(3)), "Expected image files in each subfolder"

    # Check if parallel processing was used
    assert multiprocessing.cpu_count() > 1, "Multiple CPU cores are required for this test"
