"""The package version and the CITATION.cff version must agree (the release workflow checks the tag)."""
from __future__ import annotations

import re
from pathlib import Path

import gradsolve


def test_package_version_matches_citation_file():
    cff = (Path(__file__).resolve().parent.parent / "CITATION.cff").read_text(encoding="utf-8")
    m = re.search(r"^version:\s*[\"']?([^\s\"']+)", cff, flags=re.MULTILINE)
    assert m is not None, "no top-level `version:` in CITATION.cff"
    assert m.group(1) == gradsolve.__version__
