# Contributing

Thanks for helping make sim2real-av-bias-audit better!

1. **Set up**: Create a virtualenv, `pip install -e .[dev]`, and run `pytest -q` to ensure the baseline is green.
2. **Coding style**: Favor small, composable modules and keep numpy/pandas operations vectorized. Add succinct comments when calculations are not obvious.
3. **Docs**: Update `README.md`, `docs/README_RUN.md`, and the sample report/figures if your change alters outputs.
4. **Tests**: Extend the existing pytest suite (or add new tests) for every bug fix or feature. Re-run `pytest -q` before submitting a PR.
5. **PR checklist**: Describe the gap you are closing, attach relevant plots or report snippets, and mention any schema changes explicitly.

Questions? Open an issue and tag @waymo-research.
