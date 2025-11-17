# Contributing to OpenIMC

Thank you for your interest in contributing to OpenIMC! This document provides guidelines and instructions for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone <your-fork-url>`
3. Create a branch for your changes: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Test your changes
6. Submit a pull request

## Development Setup

See the [README.md](README.md) for installation instructions. For development, you may want to install the package in editable mode:

```bash
pip install -e .
```

## Running Tests

Tests can be run using the test script:

```bash
./scripts/run_tests.sh
```

Or directly with pytest:

```bash
pytest tests/
```

## Code Style

Please follow Python PEP 8 style guidelines. The project uses type hints where appropriate.

## Submitting Changes

1. Ensure all tests pass
2. Update documentation if needed
3. Write clear commit messages
4. Submit a pull request with a clear description of your changes

## License

By contributing, you agree that your contributions will be licensed under the GNU General Public License v3.0.

