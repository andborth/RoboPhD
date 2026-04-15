# Contributing to RoboPhD

Thank you for your interest in contributing to RoboPhD!

## Ways to Contribute

### 1. Report Issues

Found a bug or have a feature request? Open an issue on GitHub with:
- Clear description of the problem or suggestion
- Steps to reproduce (for bugs)
- Your environment (Python version, OS, etc.)

### 2. Submit Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Run tests: `python -m pytest tests/`
5. Commit with clear messages
6. Push and open a pull request

### 3. Add a New Domain

Each domain is a self-contained directory under `examples/`. To add a new one:

```
examples/my_domain/
├── main.py             # Entry point using optimize_anything()
├── evaluator.py        # Domain evaluator: (candidate, example) -> (score, diagnostics)
├── objective.md        # One-line optimization goal
├── background.md       # Domain knowledge for the evolution AI
├── README.md           # Setup and usage instructions
└── seeds/baseline/     # Seed agent files (e.g., agent.py)
```

Key requirements:
- The evaluator must be thread-safe (called concurrently)
- Return rich string diagnostics (`.md` files) so the evolution AI can learn from failures
- Include `print()` calls in the seed agent for self-instrumenting diagnostics
- See existing examples for patterns

### 4. Create New Evolution Strategies

Evolution strategies live in `RoboPhD/evolution_strategies/`. Each is a directory containing:

```
my_strategy/
├── strategy.md          # Instructions for the evolution AI
```

See existing strategies (`use_your_judgment`, `data_focus`, `refinement`, `cross_pollination`) for examples.

### 5. Share Results

- Report evolution results on new or existing benchmarks
- Share successful configurations
- Contribute new evolution strategies

## Development Setup

```bash
git clone https://github.com/yourusername/RoboPhD.git
cd RoboPhD
pip install -r requirements.txt

# Quick test
python examples/cant_be_late/main.py --num-iterations 2 --evaluation-budget 60
```

## Code Style

- Follow PEP 8 guidelines
- Use type hints where practical
- Write clear docstrings for public functions
- Keep functions focused and reasonably sized

## Pull Request Guidelines

- Keep PRs focused on a single change
- Update documentation if needed
- Add tests for new functionality
- Ensure all tests pass
- Write clear PR descriptions

## Questions?

Open a discussion on GitHub or reach out to the maintainers.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
