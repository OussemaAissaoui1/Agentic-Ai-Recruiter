# Tests Directory

Test suites for quality assurance.

## Structure

```
tests/
├── unit/          # Unit tests for individual components
├── integration/   # Integration tests across components
└── e2e/           # End-to-end interview flow tests
```

## Test Categories

| Category | Purpose | Framework |
|----------|---------|-----------|
| `unit/` | Test individual functions/classes | pytest |
| `integration/` | Test agent interactions | pytest + testcontainers |
| `e2e/` | Full interview flow tests | pytest + async |

## Running Tests

```bash
# Run all tests
pytest tests/

# Run unit tests only
pytest tests/unit/

# Run with coverage
pytest tests/ --cov=agents --cov=core --cov=ml
```
