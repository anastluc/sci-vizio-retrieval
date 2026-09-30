import pymupdf
import pytest
from PIL import Image

from sci_vizio_retrieval.extractor import PDFProcessor
from tests.conftest import make_png


def make_pdf(path, text="Attention is all you need", with_image=True):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    if with_image:
        page.insert_image(pymupdf.Rect(72, 100, 272, 250), stream=make_png(size=(64, 48), mode="RGB", color=(0, 90, 200)))
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def workspace(tmp_path):
    pdfs = tmp_path / "pdfs"
    pdfs.mkdir()
    make_pdf(pdfs / "1903.00780.pdf")
    return tmp_path


def make_extractor(ws, **kwargs):
    return PDFProcessor(input_dir=ws / "pdfs", output_dir=ws / "out", db_path=str(ws / "state.db"), **kwargs)


def test_extracts_text_and_images(workspace):
    stats = make_extractor(workspace).process_directory()

    assert stats["processed_pdfs"] == 1 and stats["total_images"] == 1 and stats["failed_pdfs"] == 0
    text = (workspace / "out" / "extracted_text" / "1903.00780.txt").read_text()
    assert "Attention is all you need" in text

    images = list((workspace / "out" / "extracted_images" / "1903.00780").iterdir())
    assert len(images) == 1
    with Image.open(images[0]) as img:
        assert img.size == (64, 48)


def test_already_processed_pdfs_are_skipped(workspace):
    make_extractor(workspace).process_directory()
    stats = make_extractor(workspace).process_directory()
    assert stats["skipped_pdfs"] == 1 and stats["processed_pdfs"] == 0


def test_duplicates_are_skipped_but_kept_by_default(workspace):
    make_extractor(workspace).process_directory()
    copy = workspace / "pdfs" / "copy.pdf"
    copy.write_bytes((workspace / "pdfs" / "1903.00780.pdf").read_bytes())

    stats = make_extractor(workspace).process_directory()
    assert stats["duplicate_pdfs"] == 1
    assert copy.exists()

    # Recorded, so the next run skips it without re-hashing
    assert make_extractor(workspace).is_processed(copy)[1]["is_duplicate"]


def test_duplicates_deleted_when_enabled(workspace):
    make_extractor(workspace).process_directory()
    copy = workspace / "pdfs" / "copy.pdf"
    copy.write_bytes((workspace / "pdfs" / "1903.00780.pdf").read_bytes())

    make_extractor(workspace, delete_duplicates=True).process_directory()
    assert not copy.exists()


def test_corrupt_pdf_is_reported(workspace):
    (workspace / "pdfs" / "broken.pdf").write_bytes(b"not a pdf")
    stats = make_extractor(workspace).process_directory()
    assert stats["failed_pdfs"] == 1 and stats["processed_pdfs"] == 1


def test_transparent_images_are_flattened_onto_white(tmp_path):
    path = tmp_path / "t.png"
    path.write_bytes(make_png(color=(0, 0, 0, 0)))  # fully transparent

    PDFProcessor._optimize_image(path)

    with Image.open(path) as img:
        assert img.mode == "RGB"
        assert img.getpixel((0, 0)) == (255, 255, 255)


def test_extract_from_bytes_filters_small_images(tmp_path):
    from sci_vizio_retrieval.extractor import extract_pdf_images

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_image(pymupdf.Rect(72, 72, 372, 272), stream=make_png(size=(300, 200), mode="RGB"))
    page.insert_image(pymupdf.Rect(400, 72, 420, 92), stream=make_png(size=(16, 16), mode="RGB", color=(1, 2, 3)))
    data = doc.tobytes()

    all_images = extract_pdf_images(data)
    assert [(i.page, i.index, i.width, i.height) for i in all_images] == [(1, 1, 300, 200), (1, 2, 16, 16)]
    assert all_images[0].filename == "page1_img1.png"

    (large,) = extract_pdf_images(data, min_side=100)
    assert large.width == 300
