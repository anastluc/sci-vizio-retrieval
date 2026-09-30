import sqlite3

import pytest

import sci_vizio_retrieval.stores as stores
from sci_vizio_retrieval import cli
from sci_vizio_retrieval.processor import ImageProcessor
from tests.conftest import InMemoryStore, analysis_json


def test_parser_storage_options():
    args = cli.build_parser().parse_args(
        ["index", "--backend", "postgres", "--postgres-dsn", "postgresql://h/db",
         "--postgres-table", "figs", "--reindex"])
    assert (args.backend, args.postgres_dsn, args.postgres_table, args.reindex) == \
        ("postgres", "postgresql://h/db", "figs", True)
    assert args.func is cli.cmd_index


def test_parser_rejects_unknown_backend(capsys):
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["serve", "--backend", "mongo"])


def test_parser_extract_defaults_keep_duplicates():
    args = cli.build_parser().parse_args(["extract"])
    assert args.delete_duplicates is False


@pytest.mark.parametrize("command", ["run", "extract", "process", "index", "serve"])
def test_every_command_parses(command):
    assert cli.build_parser().parse_args([command]).command == command


def test_index_command_uses_selected_store(tmp_path, monkeypatch):
    db = tmp_path / "state.db"
    image = tmp_path / "img.png"
    image.write_bytes(b"png")
    ImageProcessor(db_path=str(db), output_dir=tmp_path / "o", vision_client=object(), embed_images=False)
    with sqlite3.connect(db) as conn:
        conn.execute("INSERT INTO image_processing (pdf_file, image_path, success_status, response) "
                     "VALUES ('p', ?, TRUE, ?)", (str(image), analysis_json("t", "d")))

    created = {}

    def fake_factory(backend, **kwargs):
        created.update(backend=backend, **kwargs)
        created["store"] = InMemoryStore()
        return created["store"]

    monkeypatch.setattr(stores, "create_vector_store", fake_factory)
    monkeypatch.setattr(cli, "setup_logging", lambda verbose: None)

    cli.main(["index", "--backend", "postgres", "--db-path", str(db), "--postgres-table", "figs"])

    assert created["backend"] == "postgres" and created["postgres_table"] == "figs"
    assert created["store"].count() == 1
    assert created["store"].closed


def test_main_exits_nonzero_on_failure(monkeypatch):
    monkeypatch.setattr(cli, "setup_logging", lambda verbose: None)
    monkeypatch.setattr(cli, "cmd_index", lambda args: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(SystemExit) as exc:
        cli.main(["index"])
    assert exc.value.code == 1
