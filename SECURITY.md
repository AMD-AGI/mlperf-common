# Security Policy

The **mlperf-common** (`primus-mllog`) maintainers and AMD take the security of
this project seriously. Thank you for helping keep it and its users safe.

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues,
pull requests, or discussions.**

Instead, use one of the private channels below:

1. **GitHub Private Vulnerability Reporting (preferred).**
   Go to the repository's **Security** tab and click
   **"Report a vulnerability"** to open a private advisory with the maintainers.
   See GitHub's guide on
   [privately reporting a security vulnerability](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability).

2. **AMD Product Security (PSIRT).**
   For AMD-wide product security concerns, you may also contact AMD PSIRT at
   **psirt@amd.com**. See the
   [AMD Product Security page](https://www.amd.com/en/resources/product-security.html)
   for more information.

### What to include
To help us triage quickly, please provide as much of the following as possible:
- A description of the vulnerability and its potential impact.
- Steps to reproduce or a proof of concept.
- Affected version(s), commit SHA, and environment
  (ROCm version, GPU, PyTorch / Megatron / Primus versions).
- Any suggested mitigation, if known.

## Our Commitment

- We will acknowledge receipt of your report within **5 business days**.
- We will provide an initial assessment and expected next steps, and keep you
  informed of progress toward a fix.
- We will coordinate the timing of any public disclosure with you and credit
  reporters who wish to be acknowledged.

Please give us a reasonable opportunity to address the issue before any public
disclosure.

## Supported Versions

This project is distributed primarily as a source package installed from the
`main` branch or a pinned commit. Security fixes are applied to the latest
`main`; we recommend always tracking the most recent commit.

| Version        | Supported |
| -------------- | :-------: |
| `main` (latest) | ✅        |
| older commits   | ❌        |
