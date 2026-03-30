# REST API Definitions

OpenAPI specifications for REST endpoints.

## Key Files

| File | Purpose |
|------|---------|
| `openapi.yaml` | Main OpenAPI 3.0 specification |
| `schemas/` | Reusable JSON Schema components |
| `examples/` | Request/response examples |

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/interviews` | Create new interview session |
| GET | `/interviews/{id}` | Get interview status |
| GET | `/interviews/{id}/report` | Get interview report |
| DELETE | `/interviews/{id}` | Cancel interview |
| GET | `/health` | Health check |

## Usage

```bash
# Generate client SDK
openapi-generator generate -i openapi.yaml -g python -o clients/python
```
