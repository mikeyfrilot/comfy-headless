# P1 — Document Python API and integration patterns

**Labels:** `docs`, `P1`

## Summary

Developers integrating Comfy Headless need API documentation.

## Problem

Without API docs:
- Integration is trial-and-error
- Return structures are unknown
- Error handling is guesswork

## Acceptance Criteria

- [ ] Python invocation example
- [ ] API return structure explained
- [ ] Error handling patterns documented
- [ ] Integration notes for agents/servers

## Suggested Content

### Python API

```python
from comfy_headless import ComfyRunner

runner = ComfyRunner()

result = runner.run(
    workflow="workflow.json",
    inputs={"prompt": "a sunset over mountains"},
    output_dir="./outputs"
)

print(result.status)      # "success" | "error"
print(result.outputs)     # ["outputs/image_001.png"]
print(result.duration)    # 12.5 (seconds)
```

### Return Structure

```python
@dataclass
class RunResult:
    status: str
    outputs: List[str]
    duration: float
    errors: Optional[List[str]]
```

### Error Handling

```python
try:
    result = runner.run(...)
except WorkflowValidationError as e:
    # Invalid graph structure
except NodeExecutionError as e:
    # Node failed during execution
```

## Location

`docs/api.md`
