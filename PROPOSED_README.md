# Comfy Headless

Run ComfyUI workflows **without the UI** — programmatically, reproducibly, and at scale.

Comfy Headless provides a headless execution layer for ComfyUI graphs so workflows can run in servers, agents, and pipelines without launching the web interface.

---

## What problem does this solve?

ComfyUI is powerful, but difficult to operationalize:

- UI-driven workflows are hard to automate
- running headless requires custom glue
- execution state is tightly coupled to the frontend
- reproducibility across runs is inconsistent
- integration with agents and APIs is awkward

Comfy Headless decouples **graph execution** from the UI.

---

## Core capabilities

- Headless execution of ComfyUI workflows
- JSON graph loading and validation
- Deterministic node execution
- Explicit input/output handling
- Programmatic graph invocation
- CLI and Python integration paths

Comfy Headless does **not** replace ComfyUI — it runs its graphs.

---

## Quick start

```bash
pip install comfy-headless

comfy-headless run workflow.json \
  --inputs inputs.json \
  --output ./outputs
```

(Commands illustrative — see repo for exact CLI/API.)

---

## Typical workflows

- Batch image generation
- Server-side rendering pipelines
- Agent-driven image workflows
- Scheduled or queued execution
- CI image validation

---

## When to use Comfy Headless

- You want ComfyUI without a browser
- You need deterministic, scriptable runs
- You are integrating with agents or APIs
- You want repeatable workflows

## When not to use it

- Interactive prompt tweaking
- Visual node graph editing
- Manual experimentation

Use standard ComfyUI for those.

---

## Design goals

- **UI-independent execution**
- **Deterministic behavior**
- **Explicit inputs and outputs**
- **Minimal ComfyUI patching**
- **Compatibility with existing graphs**

---

## Project status

**Beta**

Core execution is stable.
Compatibility with new ComfyUI nodes may evolve.

---

## Ecosystem

Part of the [MCP Tool Shop](https://github.com/mcp-tool-shop) ecosystem.

Pairs well with:

- **Dev Brain** (orchestration)
- **Tool Scan** (workflow inspection)
- **File Compass** (graph navigation)
- **Context Window Manager** (prompt shaping)

---

## License

MIT
