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

### 3. Improve Documentation

- Fix typos or clarify explanations
- Add examples or tutorials
- Translate documentation

### 4. Share Results

- Report your evolution results
- Share successful agent configurations
- Contribute new evolution strategies

## Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/RoboPhD.git
cd RoboPhD

# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/
```

## Code Style

- Follow PEP 8 guidelines
- Use type hints where practical
- Write clear docstrings for public functions
- Keep functions focused and reasonably sized

## Creating New Evolution Strategies

Evolution strategies live in `RoboPhD/evolution_strategies/`. Each strategy is a directory containing:

```
my_new_strategy/
├── strategy.md          # Strategy instructions for the evolution AI
└── strategy_tools/      # Optional helper scripts
    └── analyze_errors.py
```

See existing strategies for examples.

## Creating New Agents

Agents live in `RoboPhD/agents/`. Each agent is a directory containing:

```
my_agent/
├── agent.md             # Database analysis instructions
├── eval_instructions.md # SQL generation instructions
└── tools/               # Optional analysis scripts
    └── schema_analyzer.py
```

### Tool-Only Agents

For deterministic agents, add YAML frontmatter to `agent.md`:

```yaml
---
execution_mode: tool_only
tool_command: python tools/my_analyzer.py
tool_output_file: tool_output/analysis.txt
---
```

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
