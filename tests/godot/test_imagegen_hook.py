"""The restyle hook: gated on --instruct, resilient per asset."""

from pathlib import Path

from v2g.godot import generator as G
from v2g.llm import image_gen
from v2g.llm.analyzer import Character, GameDesign, GameObject, SceneDesign


def _design() -> GameDesign:
    return GameDesign(
        title="T",
        genre="visual novel",
        summary="s",
        narrative="",
        mechanics=[],
        controls=[],
        style="s",
        characters=[
            Character(name="Lady / The Bride", role="protagonist", visual="white gown, silver hair")
        ],
        objects=[GameObject(name="Signet Ring", role="collectible", visual="gold ring")],
        scenes=[SceneDesign(name="Courtyard", description="moonlit stone courtyard")],
    )


def _assets(tmp_path: Path) -> dict[str, Path]:
    bg = tmp_path / "background.png"
    bg.write_bytes(b"orig-bg")
    char = tmp_path / "char_lady___the_bride.png"
    char.write_bytes(b"orig-char")
    return {"background": bg, "characters/lady___the_bride": char}


class _Recording:
    def __init__(self, *, fail_on: str = "", available: bool = True):
        self.calls: list[tuple[str, str, str]] = []
        self.fail_on = fail_on
        self.available = available

    def is_available(self) -> bool:
        return self.available

    def transform(self, path: Path, prompt: str, *, reference: str = "", size: str = "") -> Path:
        if self.fail_on and self.fail_on in path.name:
            raise RuntimeError("boom")
        self.calls.append((path.name, prompt, reference))
        return path


def test_restyle_is_gated_on_instruct_and_assets(monkeypatch, tmp_path):
    provider = _Recording()
    monkeypatch.setattr(image_gen, "get_provider", lambda: provider)
    G._restyle_assets(_design(), _assets(tmp_path), None)
    G._restyle_assets(_design(), _assets(tmp_path), "")
    G._restyle_assets(_design(), {}, "vampire style")
    assert provider.calls == []


def test_restyle_passes_instruct_and_design_reference(monkeypatch, tmp_path):
    provider = _Recording()
    monkeypatch.setattr(image_gen, "get_provider", lambda: provider)
    assets = _assets(tmp_path)

    G._restyle_assets(_design(), assets, "vampire style")

    refs = {name: reference for name, _, reference in provider.calls}
    assert refs["char_lady___the_bride.png"] == "Lady / The Bride, white gown, silver hair"
    assert refs["background.png"] == "Courtyard: moonlit stone courtyard"
    assert all(prompt == "vampire style" for _, prompt, _ in provider.calls)


def test_restyle_keeps_originals_when_provider_unavailable(monkeypatch, tmp_path):
    provider = _Recording(available=False)
    monkeypatch.setattr(image_gen, "get_provider", lambda: provider)
    assets = _assets(tmp_path)

    G._restyle_assets(_design(), assets, "vampire style")

    assert provider.calls == []
    assert assets["background"].read_bytes() == b"orig-bg"


def test_restyle_keeps_the_original_of_a_failing_asset(monkeypatch, tmp_path):
    provider = _Recording(fail_on="char_")
    monkeypatch.setattr(image_gen, "get_provider", lambda: provider)
    assets = _assets(tmp_path)
    char = assets["characters/lady___the_bride"]

    G._restyle_assets(_design(), assets, "vampire style")

    assert assets["characters/lady___the_bride"] == char  # unchanged on failure
    assert assets["background"].read_bytes() == b"orig-bg"  # untouched without a real model
    assert [name for name, _, _ in provider.calls] == ["background.png"]  # others proceeded
