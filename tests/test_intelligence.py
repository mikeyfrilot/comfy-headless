"""Tests for intelligence module."""



class TestPromptCache:
    """Test PromptCache LRU implementation."""

    def test_cache_set_and_get(self):
        """Test basic cache operations."""
        from comfy_headless.intelligence import PromptAnalysis, PromptCache

        cache = PromptCache(max_size=10)

        analysis = PromptAnalysis(
            original="test prompt",
            intent="test",
            styles=["style1"],
            mood="neutral",
            suggested_preset="quality",
        )
        cache.set_analysis("test prompt", analysis)

        result = cache.get_analysis("test prompt")
        assert result is not None
        assert result.intent == "test"

    def test_cache_miss(self):
        """Test cache miss returns None."""
        from comfy_headless.intelligence import PromptCache

        cache = PromptCache(max_size=10)
        result = cache.get_analysis("nonexistent")
        assert result is None

    def test_cache_eviction(self):
        """Test LRU eviction when cache is full."""
        from comfy_headless.intelligence import PromptAnalysis, PromptCache

        cache = PromptCache(max_size=3)

        # Fill cache
        for i in range(5):
            analysis = PromptAnalysis(
                original=f"prompt{i}",
                intent=f"intent{i}",
                styles=[],
                mood="neutral",
                suggested_preset="quality",
            )
            cache.set_analysis(f"prompt{i}", analysis)

        # Early entries should be evicted
        assert cache.get_analysis("prompt0") is None
        assert cache.get_analysis("prompt1") is None
        # Recent entries should exist
        assert cache.get_analysis("prompt4") is not None

    def test_cache_stats(self):
        """Test cache statistics."""
        from comfy_headless.intelligence import PromptAnalysis, PromptCache

        cache = PromptCache(max_size=10)

        analysis = PromptAnalysis(
            original="test", intent="test", styles=[], mood="neutral", suggested_preset="quality"
        )
        cache.set_analysis("test", analysis)

        stats = cache.stats()
        assert "analysis_entries" in stats
        assert stats["analysis_entries"] == 1


class TestSanitizePrompt:
    """Test prompt sanitization."""

    def test_sanitize_removes_control_chars(self):
        """Test control characters are removed."""
        from comfy_headless.intelligence import sanitize_prompt

        result = sanitize_prompt("test\x00prompt\x1f")
        assert "\x00" not in result
        assert "\x1f" not in result
        assert "testprompt" in result

    def test_sanitize_truncates_long_prompts(self):
        """Test long prompts are truncated."""
        from comfy_headless.intelligence import sanitize_prompt

        long_prompt = "a" * 3000
        result = sanitize_prompt(long_prompt, max_length=100)
        assert len(result) == 100

    def test_sanitize_handles_empty(self):
        """Test empty prompt handling."""
        from comfy_headless.intelligence import sanitize_prompt

        assert sanitize_prompt("") == ""
        assert sanitize_prompt(None) == ""


class TestFewShotExamples:
    """Test few-shot example system."""

    def test_builtin_examples_exist(self):
        """Test built-in examples are present."""
        from comfy_headless.intelligence import FEW_SHOT_ENHANCEMENT_EXAMPLES

        assert len(FEW_SHOT_ENHANCEMENT_EXAMPLES) > 0
        assert all("input" in ex for ex in FEW_SHOT_ENHANCEMENT_EXAMPLES)
        assert all("output" in ex for ex in FEW_SHOT_ENHANCEMENT_EXAMPLES)

    def test_get_few_shot_prompt(self):
        """Test few-shot prompt generation."""
        from comfy_headless.intelligence import get_few_shot_prompt

        prompt = get_few_shot_prompt("balanced")
        assert "Input:" in prompt
        assert "Output:" in prompt

    def test_get_few_shot_examples(self):
        """Test get_few_shot_examples returns list."""
        from comfy_headless.intelligence import get_few_shot_examples

        examples = get_few_shot_examples()
        assert isinstance(examples, list)
        assert len(examples) > 0


class TestPromptAnalysis:
    """Test PromptAnalysis dataclass."""

    def test_prompt_analysis_creation(self):
        """Test PromptAnalysis creation."""
        from comfy_headless.intelligence import PromptAnalysis

        analysis = PromptAnalysis(
            original="portrait of a woman",
            intent="portrait",
            styles=["realistic", "cinematic"],
            mood="dramatic",
            suggested_preset="quality",
        )

        assert analysis.intent == "portrait"
        assert "realistic" in analysis.styles
        assert analysis.mood == "dramatic"
        assert analysis.original == "portrait of a woman"


class TestEnhancedPrompt:
    """Test EnhancedPrompt dataclass."""

    def test_enhanced_prompt_creation(self):
        """Test EnhancedPrompt creation."""
        from comfy_headless.intelligence import EnhancedPrompt

        enhanced = EnhancedPrompt(
            original="cat",
            enhanced="a fluffy orange cat sitting on a windowsill",
            negative="blurry, distorted",
            additions=["fluffy", "orange", "sitting", "windowsill"],
            reasoning="Added descriptive details",
        )

        assert enhanced.original == "cat"
        assert "fluffy" in enhanced.enhanced
        assert "blurry" in enhanced.negative
        assert len(enhanced.additions) == 4
