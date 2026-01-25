# P2 — Add performance and scaling guidance

**Labels:** `docs`, `performance`, `P2`

## Summary

Users running at scale need performance guidance.

## Problem

Without scaling docs:
- GPU memory issues are mysterious
- Batch strategies are unclear
- Parallelism is trial-and-error

## Acceptance Criteria

- [ ] GPU memory notes (VRAM requirements by model)
- [ ] Batch execution tips
- [ ] Parallelism guidance (when safe, when not)
- [ ] Queue/worker patterns for high-volume

## Suggested Content

### GPU Memory

| Model Type | Minimum VRAM | Recommended |
|------------|--------------|-------------|
| SD 1.5     | 4GB          | 6GB         |
| SDXL       | 8GB          | 12GB        |
| Flux       | 12GB         | 16GB+       |

### Batch Execution

```bash
# Sequential (safe, predictable)
comfy-headless batch workflow.json --prompts prompts.txt

# Memory-efficient batching
comfy-headless batch workflow.json --batch-size 4 --clear-cache
```

### Parallelism

- **Safe**: Multiple workflows on separate GPUs
- **Unsafe**: Multiple workflows on same GPU (OOM risk)
- **Recommended**: Queue-based worker pattern

### Queue Pattern

```python
# Worker consuming from queue
while job := queue.get():
    result = runner.run(job.workflow, job.inputs)
    queue.complete(job.id, result)
```

## Location

`docs/performance.md`
