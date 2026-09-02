"""Sphinx configuration for the documentation site (Furo theme, Markdown via MyST)."""

project = "gradsolve"
author = "Alessio Spurio Mancini"
copyright = "2026, Alessio Spurio Mancini"
extensions = ["myst_parser"]
root_doc = "contents"
exclude_patterns = ["_build", "assets/logo/README.md"]
myst_heading_anchors = 3
html_theme = "furo"
html_title = "gradsolve"
html_static_path = ["assets"]
html_favicon = "assets/logo/favicon.svg"
html_theme_options = {
    "light_logo": "logo/gradsolve-lockup-light.svg",
    "dark_logo": "logo/gradsolve-lockup-dark.svg",
    "sidebar_hide_name": True,
    "source_repository": "https://github.com/ECLIPSE-AI4Science/gradsolve",
    "source_branch": "main",
    "source_directory": "docs/",
}
