# P0 — Add minimal headless quickstart

**Labels:** `docs`, `P0`

## Summary

Users need a copy-paste example that runs a workflow headlessly in under 2 minutes.

## Problem

Without a quickstart:
- Users don't know where to start
- Existing workflow reuse is unclear
- Time-to-value is too high

## Acceptance Criteria

- [ ] Copy-paste runnable example
- [ ] Example workflow JSON included (or linked)
- [ ] Output directory shown with expected results
- [ ] Prerequisites clearly stated (ComfyUI models, etc.)

## Suggested Structure

```markdown
## Quickstart

### Prerequisites
- ComfyUI models installed at `~/.comfyui/models/`
- Python 3.10+

### Install
pip install comfy-headless

### Run a workflow
comfy-headless run examples/txt2img.json --output ./outputs

### Check results
ls ./outputs/
# image_001.png
```

## Notes

The quickstart should:
- Use a simple text-to-image workflow
- Complete in reasonable time (model-dependent)
- Produce a visible output file
- Not require any configuration beyond install
