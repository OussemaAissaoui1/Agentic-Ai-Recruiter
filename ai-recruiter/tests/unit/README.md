# Unit Tests

Unit tests for individual components.

## Structure

```
unit/
├── agents/         # Tests for each agent
│   ├── test_orchestrator.py
│   ├── test_nlp.py
│   ├── test_vision.py
│   ├── test_voice.py
│   ├── test_avatar.py
│   └── test_scoring.py
├── core/           # Tests for core modules
│   ├── test_memory.py
│   ├── test_messaging.py
│   └── test_state.py
└── ml/             # Tests for ML pipeline
    ├── test_preprocessing.py
    └── test_evaluation.py
```

## Guidelines

- Test one function/method per test
- Mock external dependencies
- Use fixtures for common setup
- Aim for >80% coverage
