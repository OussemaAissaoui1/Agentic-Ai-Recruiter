# gRPC API Definitions

Protocol Buffer definitions for gRPC services.

## Key Files

| File | Purpose |
|------|---------|
| `interview.proto` | Interview service (start, stop, status) |
| `streaming.proto` | Bidirectional streaming messages |
| `scoring.proto` | Scoring service definitions |
| `health.proto` | Health check service |

## Services

```protobuf
service InterviewService {
    rpc StartInterview(StartRequest) returns (InterviewSession);
    rpc EndInterview(EndRequest) returns (InterviewReport);
    rpc GetStatus(StatusRequest) returns (InterviewStatus);
    rpc StreamInterview(stream ClientMessage) returns (stream ServerMessage);
}
```

## Code Generation

```bash
# Generate Python stubs
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. *.proto
```
