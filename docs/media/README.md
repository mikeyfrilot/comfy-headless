# Media Assets Guidance

## Recommended Visuals

Comfy Headless is best explained through diagrams showing the execution flow.

### Priority assets to create:

1. **ComfyUI UI vs Headless Execution**
   - Side-by-side comparison
   - UI path: browser → graph → execute
   - Headless path: JSON → comfy-headless → outputs

2. **Workflow Execution Flow**
   ```
   workflow.json → validation → execution → outputs/
   ```
   - Show inputs entering
   - Show outputs emerging
   - Emphasize no browser required

3. **Example Output Grid**
   - Show sample generated images
   - Demonstrate batch capability
   - Visual proof it works

## Format Recommendations

| Format | Use Case |
|--------|----------|
| SVG | Architecture diagrams |
| PNG | Output image examples |
| GIF | Terminal execution demos |

## Diagram Style

- Keep diagrams simple and clean
- Use two-color schemes (headless vs UI)
- Show the "before/after" of removing the UI dependency

## Why Diagrams Matter

Users need to see:
- That their existing workflows still work
- That the UI is truly optional
- That outputs are identical

A simple flow diagram communicates this faster than text.
