from types import SimpleNamespace

import pytest

import sci_vizio_retrieval.client as client_module
from sci_vizio_retrieval.client import OpenRouterVision


class FakeCompletions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=outcome))])


def fake_client(*outcomes):
    completions = FakeCompletions(outcomes)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(client_module.time, "sleep", lambda s: None)


def test_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY")
    monkeypatch.setattr(client_module, "load_dotenv", lambda: None)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        OpenRouterVision()


def test_builds_openrouter_client():
    vision = OpenRouterVision(model="some/model")
    assert str(vision.client.base_url).startswith("https://openrouter.ai/api/v1")
    assert vision.model == "some/model"


@pytest.mark.parametrize("name, mime", [
    ("a.png", "image/png"), ("a.JPG", "image/jpeg"), ("a.jpeg", "image/jpeg"), ("a.unknownext", "image/jpeg"),
])
def test_mime_type(name, mime):
    assert OpenRouterVision._mime_type(name) == mime


def test_analyze_image_sends_image_and_prompt(image_file):
    api, completions = fake_client('{"ok": true}')
    vision = OpenRouterVision(model="m", api_delay=0, max_tokens=123, client=api)

    assert vision.analyze_image(image_file, "describe").text == '{"ok": true}'

    (call,) = completions.calls
    assert call["model"] == "m" and call["max_tokens"] == 123
    text_part, image_part = call["messages"][0]["content"]
    assert text_part == {"type": "text", "text": "describe"}
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")


def test_retries_then_succeeds(image_file):
    api, completions = fake_client(RuntimeError("503"), "done")
    vision = OpenRouterVision(api_delay=0, max_retries=3, client=api)
    assert vision.analyze_image(image_file, "p").text == "done"
    assert len(completions.calls) == 2


def test_raises_after_exhausting_retries(image_file):
    api, completions = fake_client(RuntimeError("a"), RuntimeError("b"))
    vision = OpenRouterVision(api_delay=0, max_retries=2, client=api)
    with pytest.raises(RuntimeError, match="b"):
        vision.analyze_image(image_file, "p")
    assert len(completions.calls) == 2


def test_zero_retries_still_makes_one_attempt(image_file):
    api, _ = fake_client("x")
    assert OpenRouterVision(api_delay=0, max_retries=0, client=api).analyze_image(image_file, "p").text == "x"
