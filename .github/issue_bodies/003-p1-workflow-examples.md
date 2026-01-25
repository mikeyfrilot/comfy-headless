# P1 — Add common workflow examples

**Labels:** `docs`, `P1`

## Summary

Users need example workflows for common use cases.

## Problem

Without examples:
- Users copy from ComfyUI forums blindly
- Best practices are unclear
- Agent integration patterns are unknown

## Acceptance Criteria

- [ ] Text-to-image example workflow
- [ ] Batch rendering example
- [ ] Agent-driven example (with input substitution)
- [ ] Each example includes JSON and explanation

## Suggested Examples

### 1. Basic Text-to-Image
```json
// txt2img.json - simplest possible workflow
```

### 2. Batch Rendering
```bash
# Run same workflow with different prompts
comfy-headless batch workflow.json --prompts prompts.txt
```

### 3. Agent Integration
```python
from comfy_headless import run_workflow

result = run_workflow(
    "workflow.json",
    inputs={"prompt": agent_generated_prompt}
)
```

## Location

`examples/` directory with README explaining each
