# Contributing to gradsolve

Thanks for your interest in improving gradsolve. This guide is deliberately short.

## Development setup

gradsolve targets Python 3.11–3.13. Install it in editable mode with the test
extra:

```bash
pip install -e '.[test]'
```

conda users:

```bash
conda create -n gradsolve python=3.12
conda activate gradsolve
pip install -e '.[test]'
```

gradsolve enables float64 on import; set `GRADSOLVE_X64=0` before importing if you
need to opt out.

## The commands that must pass

Run these before opening a pull request — CI runs the same checks:

```bash
pytest -q                    # unit tests (CPU)
ruff check gradsolve tests     # lint
python -m build              # build gate: sdist + wheel must build cleanly
```

For the build gate you also need `build` and `twine` (`pip install -e '.[release]'`
installs the pinned tooling); `twine check dist/*` should report `PASSED`.

## Pull request expectations

- Keep changes focused; one logical change per PR.
- Add or update tests for any behavior change; `pytest -q` stays green.
- `ruff check gradsolve tests` is clean (no new lint errors).
- Update [`CHANGELOG.md`](CHANGELOG.md) (Keep a Changelog format, newest first)
  in the same PR as the change it describes.
- Public-facing behavior changes come with a matching docs update under `docs/`.
- CI runs the Warp kernels on their CPU backend; nothing in CI runs on a GPU. Validate GPU
  execution (`gradsolve[cuda12]`, or `gradsolve[warp]` on a CUDA device) locally and note in
  the PR how you did.

## License of contributions

By contributing you agree that your contributions are licensed under the
project's [MIT License](LICENSE).
