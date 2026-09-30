import json
import sqlite3

import pytest

from sci_vizio_retrieval.client import VisionResponse
from sci_vizio_retrieval.processor import ImageProcessor, ImageProcessorRetry, sanitize_filename
from tests.conftest import analysis_json, make_png


class FakeVision:
    def __init__(self, fail_on=()):
        self.calls = []
        self.fail_on = set(fail_on)

    def analyze_image(self, image_path, prompt):
        self.calls.append(str(image_path))
        if image_path.name in self.fail_on:
            raise RuntimeError("model overloaded")
        return VisionResponse(text=analysis_json(f"Figure {image_path.stem}", "desc"))


@pytest.fixture
def images_dir(tmp_path):
    root = tmp_path / "extracted_images"
    for pdf in ("paperA", "paperB"):
        (root / pdf).mkdir(parents=True)
        (root / pdf / "page1_img1.png").write_bytes(make_png())
    (root / "paperA" / "notes.txt").write_text("not an image")
    return root


def make_processor(tmp_path, images_dir, vision):
    return ImageProcessor(images_dir=str(images_dir), output_dir=str(tmp_path / "analysis"),
                          db_path=str(tmp_path / "state.db"), vision_client=vision, embed_images=False)


def test_sanitize_filename():
    assert sanitize_filename('a/b:c*d?"e<f>g|h') == "a_b_c_d__e_f_g_h"


def test_process_directory_writes_results(tmp_path, images_dir):
    vision = FakeVision()
    stats = make_processor(tmp_path, images_dir, vision).process_directory()

    assert stats == {"total_images": 2, "successful": 2, "failed": 0, "cached": 0}
    out = tmp_path / "analysis" / "paperA" / "page1_img1_analysis.json"
    assert json.loads(out.read_text())["title"] == "Figure page1_img1"

    with sqlite3.connect(tmp_path / "state.db") as conn:
        rows = conn.execute("SELECT pdf_file, success_status, response_status_code, embedding "
                            "FROM image_processing ORDER BY pdf_file").fetchall()
    assert rows == [("paperA", 1, 200, None), ("paperB", 1, 200, None)]


def test_results_are_cached(tmp_path, images_dir):
    make_processor(tmp_path, images_dir, FakeVision()).process_directory()
    (tmp_path / "analysis" / "paperA" / "page1_img1_analysis.json").unlink()

    vision = FakeVision()
    stats = make_processor(tmp_path, images_dir, vision).process_directory()

    assert stats["cached"] == 2 and vision.calls == []
    # Cached responses restore missing analysis files
    assert (tmp_path / "analysis" / "paperA" / "page1_img1_analysis.json").exists()


def test_pdf_filter(tmp_path, images_dir):
    vision = FakeVision()
    stats = make_processor(tmp_path, images_dir, vision).process_directory(pdf_names=["paperB"])
    assert stats["total_images"] == 1
    assert "paperB" in vision.calls[0]


def test_failures_recorded_then_retried(tmp_path, images_dir):
    make_processor(tmp_path, images_dir, FakeVision(fail_on={"page1_img1.png"})).process_directory()

    with sqlite3.connect(tmp_path / "state.db") as conn:
        errors = conn.execute("SELECT error_message FROM image_processing WHERE success_status = FALSE").fetchall()
    assert len(errors) == 2 and "model overloaded" in errors[0][0]

    retry = ImageProcessorRetry(images_dir=str(images_dir), output_dir=str(tmp_path / "analysis"),
                                db_path=str(tmp_path / "state.db"), vision_client=FakeVision())
    assert retry.process_failed_entries() == {"total_entries": 2, "successful": 2, "failed": 0}
    assert retry.get_failed_entries() == []
    assert (tmp_path / "analysis" / "paperB" / "page1_img1_analysis.json").exists()


def test_retry_with_nothing_to_do(tmp_path, images_dir):
    make_processor(tmp_path, images_dir, FakeVision()).process_directory()
    retry = ImageProcessorRetry(db_path=str(tmp_path / "state.db"), output_dir=str(tmp_path / "a"),
                                vision_client=FakeVision())
    assert retry.process_failed_entries()["total_entries"] == 0


def test_missing_images_dir(tmp_path):
    stats = make_processor(tmp_path, tmp_path / "nope", FakeVision()).process_directory()
    assert stats["total_images"] == 0
