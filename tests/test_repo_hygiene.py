from __future__ import annotations

from pathlib import Path


def test_env_examples_exist_and_real_envs_are_ignored():
    root = Path(__file__).resolve().parents[1]
    gitignore = (root / ".gitignore").read_text()

    assert ".env" in gitignore
    assert "docker/env/*.env" in gitignore
    assert (root / ".env.example").exists()
    assert (root / "docker/env/.env.example").exists()


def test_env_examples_only_contain_placeholders():
    root = Path(__file__).resolve().parents[1]
    for path in [root / ".env.example", root / "docker/env/.env.example"]:
        text = path.read_text()
        assert "CHANGE_ME" in text
        assert "token" not in text.lower()
