import configparser
from pathlib import Path

import pytest

from sci_vizio_retrieval import config


@pytest.fixture
def ini(monkeypatch):
    parser = configparser.ConfigParser()
    parser.read_string("[vision]\nmodel = from-ini\nmax_tokens = nope\n")
    monkeypatch.setattr(config, "config", parser)
    return parser


def test_value_precedence(monkeypatch, ini):
    assert config.get_config_value("vision", "model", "default") == "from-ini"
    assert config.get_config_value("vision", "missing", "default") == "default"

    monkeypatch.setenv("VISION_MODEL", "from-short-env")
    assert config.get_config_value("vision", "model", "default") == "from-short-env"

    monkeypatch.setenv("SCI_VISION_MODEL", "from-prefixed-env")
    assert config.get_config_value("vision", "model", "default") == "from-prefixed-env"


def test_bare_key_env_var_is_not_used(monkeypatch, ini):
    """A generic `MODEL` variable must not hijack the vision model."""
    monkeypatch.setenv("MODEL", "unrelated")
    assert config.get_config_value("vision", "model", "default") == "from-ini"


def test_numeric_parsing_falls_back_to_default(monkeypatch, ini):
    assert config.get_config_int("vision", "max_tokens", 7) == 7
    monkeypatch.setenv("SCI_VISION_API_DELAY", "0.5")
    assert config.get_config_float("vision", "api_delay", 2.0) == 0.5


def test_resolve_path():
    assert config.resolve_path("/abs/x") == Path("/abs/x")
    assert config.resolve_path("rel/x") == config.PROJECT_ROOT / "rel/x"


def test_output_subdirs(tmp_path):
    assert config.get_images_dir(str(tmp_path)) == tmp_path / config.IMAGES_SUBDIR
    assert config.get_analysis_dir(str(tmp_path)) == tmp_path / config.ANALYSIS_SUBDIR


def test_default_config_template_is_valid():
    parser = configparser.ConfigParser()
    parser.read_string(config.DEFAULT_CONFIG_CONTENT)
    assert parser.get("storage", "backend") in config.STORAGE_BACKENDS
    assert parser.get("storage", "postgres_table") == "image_analysis"
