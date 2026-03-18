# End-to-End Tests

Full interview flow tests.

## Scope

Test complete interview scenarios:
- Interview start to finish
- Multi-turn conversations
- Error handling and recovery
- Scoring and report generation

## Test Scenarios

| Scenario | Description |
|----------|-------------|
| `test_happy_path.py` | Complete successful interview |
| `test_connection_drops.py` | Handle client disconnection |
| `test_timeout_handling.py` | Handle agent timeouts |
| `test_scoring_accuracy.py` | Verify scoring consistency |

## Running E2E Tests

```bash
# Requires all services running
docker-compose up -d
pytest tests/e2e/ -v
```
