# Contributing to mlperf-common

Thanks for your interest in contributing to **mlperf-common** (`primus-mllog`)!
This project provides MLPerf-compliant training logging utilities for ROCm Primus.
Contributions of all kinds are welcome — bug reports, documentation, and code.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Ways to Contribute](#ways-to-contribute)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Commit Sign-Off (DCO)](#commit-sign-off-dco)
- [Pull Request Process](#pull-request-process)
- [Coding Guidelines](#coding-guidelines)
- [Reporting Security Issues](#reporting-security-issues)
- [License](#license)

## Code of Conduct
Please be respectful and constructive in all interactions. We want this to be a
welcoming, harassment-free community for everyone. Report unacceptable behavior
to the maintainers listed in [`.github/CODEOWNERS`](./.github/CODEOWNERS).

## Ways to Contribute
- **Report a bug** — open a [GitHub Issue](https://github.com/AMD-AGI/mlperf-common/issues)
  with steps to reproduce, expected vs. actual behavior, and your environment
  (ROCm version, GPU, PyTorch, Megatron/Primus versions).
- **Suggest an enhancement** — open an issue describing the use case and proposed change.
- **Submit a fix or feature** — open a pull request following the process below.
- **Improve documentation** — corrections and clarifications to the README or docstrings are appreciated.

## Development Setup
This is a Python package. For local development:

```bash
git clone https://github.com/AMD-AGI/mlperf-common.git
cd mlperf-common
python -m venv .venv && source .venv/bin/activate   # optional
pip install -e .
```

Runtime features assume a ROCm-enabled PyTorch stack (AMD Instinct GPUs such as
MI300X / MI355X) plus `mlperf-logging`, Primus, and Megatron-LM / Megatron-Core.
You can develop and review most logging logic without GPUs, but end-to-end
validation requires the full training stack.

## Making Changes
1. Fork the repository (external contributors) or create a branch (maintainers).
2. Create a descriptive branch, e.g. `fix/eval-stop-timing` or `docs/update-readme`.
3. Keep changes focused and reasonably small — one logical change per pull request.
4. Update the README and docstrings when you change public behavior or configuration.

## Commit Sign-Off (DCO)
This project uses the [Developer Certificate of Origin](https://developercertificate.org/).
Every commit must be signed off to certify that you wrote the change or otherwise
have the right to submit it under the project's license:

```bash
git commit -s -m "Short, imperative summary of the change"
```

The `-s` flag adds a `Signed-off-by: Your Name <your.email@example.com>` trailer.
Use a real name and email.

## Pull Request Process
1. Ensure your branch is up to date with `main`.
2. Push your branch and open a pull request against `main`.
3. In the PR description, explain **what** changed and **why**, and link any related issue.
4. A maintainer (see [`.github/CODEOWNERS`](./.github/CODEOWNERS)) will be requested for review automatically.
5. Address review feedback and keep the PR rebased/mergeable.
6. PRs are merged once they have an approving review and any CI checks pass.

## Coding Guidelines
- Target Python and follow the style and structure already present in
  [`primus_mllog/`](./primus_mllog).
- Prefer clear, self-documenting code; add comments only where intent is non-obvious.
- Keep rank-awareness and MLPerf-logging semantics intact — spurious events or
  wrong-rank emissions can invalidate an MLPerf submission.
- New configuration should be driven by environment variables and documented in
  the README's configuration tables.

## Reporting Security Issues
Please **do not** report security vulnerabilities through public issues or pull
requests. Follow the process in [`SECURITY.md`](./SECURITY.md) instead.

## License
By contributing, you agree that your contributions will be licensed under the
[Apache License 2.0](./LICENSE), the same license that covers this project.
