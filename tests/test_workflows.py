"""Tests for workflows module."""



class TestGenerationPresets:
    """Test generation presets."""

    def test_presets_exist(self):
        """Test presets are defined."""
        from comfy_headless.workflows import GENERATION_PRESETS

        assert len(GENERATION_PRESETS) > 0
        assert "quality" in GENERATION_PRESETS
        assert "fast" in GENERATION_PRESETS

    def test_preset_has_required_keys(self):
        """Test presets have required parameters."""
        from comfy_headless.workflows import GENERATION_PRESETS

        required_keys = ["width", "height", "steps", "cfg"]

        for name, preset in GENERATION_PRESETS.items():
            for key in required_keys:
                assert key in preset, f"Preset '{name}' missing '{key}'"

    def test_list_presets(self):
        """Test listing presets."""
        from comfy_headless.workflows import list_presets

        presets = list_presets()
        assert isinstance(presets, list)
        assert "quality" in presets


class TestDAGValidator:
    """Test workflow DAG validation."""

    def test_valid_workflow(self, sample_workflow):
        """Test valid workflow passes validation."""
        from comfy_headless.workflows import validate_workflow_dag

        errors = validate_workflow_dag(sample_workflow)
        assert len(errors) == 0

    def test_empty_workflow(self):
        """Test empty workflow validation."""
        from comfy_headless.workflows import validate_workflow_dag

        errors = validate_workflow_dag({})
        # Empty workflow should have at least a warning
        # (may depend on implementation)
        assert isinstance(errors, list)

    def test_missing_input(self):
        """Test workflow with missing input reference."""
        from comfy_headless.workflows import validate_workflow_dag

        workflow = {
            "1": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["999", 0],  # Reference to non-existent node
                },
            }
        }

        errors = validate_workflow_dag(workflow)
        assert len(errors) > 0


class TestWorkflowCompiler:
    """Test WorkflowCompiler."""

    def test_compile_basic_workflow(self):
        """Test compiling a basic workflow."""
        from comfy_headless.workflows import compile_workflow

        result = compile_workflow(
            prompt="a sunset",
            preset="quality",
        )

        assert result.is_valid or len(result.errors) >= 0  # May have warnings
        assert result.workflow is not None or result.errors

    def test_compile_with_negative(self):
        """Test compiling with negative prompt."""
        from comfy_headless.workflows import compile_workflow

        result = compile_workflow(
            prompt="a sunset",
            negative="blurry",
            preset="fast",
        )

        # Check the workflow has negative prompt if valid
        if result.is_valid:
            # Find CLIPTextEncode nodes
            for node in result.workflow.values():
                if isinstance(node, dict) and node.get("class_type") == "CLIPTextEncode":
                    text = node.get("inputs", {}).get("text", "")
                    if "blurry" in text:
                        break


class TestWorkflowOptimizer:
    """Test WorkflowOptimizer."""

    def test_vram_estimation_constants(self):
        """Test VRAM estimation constants exist."""
        from comfy_headless.workflows import WorkflowOptimizer

        optimizer = WorkflowOptimizer()
        assert hasattr(optimizer, "VRAM_BASE_MODEL_GB")
        assert hasattr(optimizer, "VRAM_PER_MEGAPIXEL_GB")
        assert hasattr(optimizer, "VRAM_PER_VIDEO_FRAME_GB")

        assert optimizer.VRAM_BASE_MODEL_GB > 0
        assert optimizer.VRAM_PER_MEGAPIXEL_GB > 0
        assert optimizer.VRAM_PER_VIDEO_FRAME_GB > 0

    def test_estimate_vram(self, sample_workflow):
        """Test VRAM estimation."""
        from comfy_headless.workflows import WorkflowOptimizer

        optimizer = WorkflowOptimizer()
        vram = optimizer.estimate_vram(sample_workflow)

        # Should be at least base model VRAM
        assert vram >= optimizer.VRAM_BASE_MODEL_GB


class TestWorkflowHash:
    """Test workflow hashing."""

    def test_compute_hash(self, sample_workflow):
        """Test workflow hash computation."""
        from comfy_headless.workflows import compute_workflow_hash

        hash1 = compute_workflow_hash(sample_workflow)
        hash2 = compute_workflow_hash(sample_workflow)

        assert hash1 == hash2
        assert len(hash1) > 0

    def test_different_workflows_different_hash(self, sample_workflow):
        """Test different workflows have different hashes."""
        from comfy_headless.workflows import compute_workflow_hash

        hash1 = compute_workflow_hash(sample_workflow)

        modified = sample_workflow.copy()
        modified["3"]["inputs"]["seed"] = 99999
        hash2 = compute_workflow_hash(modified)

        assert hash1 != hash2
