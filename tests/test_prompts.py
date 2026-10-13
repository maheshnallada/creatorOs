from pathlib import Path

import pytest


def load_prompt_loader():
    try:
        from app.prompts.loader import PromptLoader
    except ModuleNotFoundError as error:
        pytest.fail(f"prompt loader implementation is missing: {error.name}")
    return PromptLoader


def test_prompt_loader_renders_declared_placeholders(tmp_path: Path):
    prompt_path = tmp_path / "linkedin" / "writer" / "system.md"
    prompt_path.parent.mkdir(parents=True)
    prompt_path.write_text("Write a {platform} post about {topic}.", encoding="utf-8")
    prompt_loader = load_prompt_loader()(tmp_path)

    rendered = prompt_loader.render(
        "linkedin.writer.system", {"platform": "LinkedIn", "topic": "AI safety"}
    )

    assert rendered == "Write a LinkedIn post about AI safety."


def test_prompt_loader_rejects_missing_or_extra_variables(tmp_path: Path):
    prompt_path = tmp_path / "linkedin" / "writer" / "system.md"
    prompt_path.parent.mkdir(parents=True)
    prompt_path.write_text("Write about {topic}.", encoding="utf-8")
    prompt_loader = load_prompt_loader()(tmp_path)

    with pytest.raises(ValueError, match="topic"):
        prompt_loader.render("linkedin.writer.system", {})
    with pytest.raises(ValueError, match="unused"):
        prompt_loader.render("linkedin.writer.system", {"topic": "AI", "unused": "value"})


def test_prompt_loader_rejects_unknown_and_unsafe_ids(tmp_path: Path):
    prompt_loader = load_prompt_loader()(tmp_path)

    with pytest.raises(FileNotFoundError):
        prompt_loader.render("linkedin.writer.system", {})
    with pytest.raises(ValueError, match="prompt ID"):
        prompt_loader.render("../../outside", {})
