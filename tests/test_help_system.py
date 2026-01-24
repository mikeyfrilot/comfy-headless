"""Tests for help_system module."""



class TestHelpLevel:
    """Test HelpLevel enum."""

    def test_help_level_values(self):
        """Test HelpLevel enum values."""
        from comfy_headless.help_system import HelpLevel

        assert HelpLevel.ELI5.value == "eli5"
        assert HelpLevel.CASUAL.value == "casual"
        assert HelpLevel.DEVELOPER.value == "developer"

    def test_help_level_from_string(self):
        """Test creating HelpLevel from string."""
        from comfy_headless.help_system import HelpLevel

        assert HelpLevel("eli5") == HelpLevel.ELI5
        assert HelpLevel("casual") == HelpLevel.CASUAL
        assert HelpLevel("developer") == HelpLevel.DEVELOPER


class TestHelpLevelManagement:
    """Test help level get/set functions."""

    def test_set_help_level(self):
        """Test setting help level."""
        from comfy_headless.help_system import HelpLevel, get_help_level, set_help_level

        original = get_help_level()

        set_help_level(HelpLevel.DEVELOPER)
        assert get_help_level() == HelpLevel.DEVELOPER

        set_help_level(HelpLevel.ELI5)
        assert get_help_level() == HelpLevel.ELI5

        # Restore original
        set_help_level(original)

    def test_get_help_level_default(self):
        """Test default help level."""
        from comfy_headless.help_system import HelpLevel, get_help_level

        level = get_help_level()
        assert isinstance(level, HelpLevel)


class TestHelpTopic:
    """Test HelpTopic dataclass."""

    def test_help_topic_creation(self):
        """Test creating HelpTopic."""
        from comfy_headless.help_system import HelpTopic

        topic = HelpTopic(
            id="test",
            title="Test Topic",
            eli5="Simple explanation",
            casual="Regular explanation",
            developer="Technical explanation",
        )

        assert topic.id == "test"
        assert topic.title == "Test Topic"

    def test_help_topic_get_content_eli5(self):
        """Test getting ELI5 content."""
        from comfy_headless.help_system import HelpLevel, HelpTopic

        topic = HelpTopic(
            id="test", title="Test", eli5="Simple", casual="Regular", developer="Technical"
        )

        assert topic.get_content(HelpLevel.ELI5) == "Simple"

    def test_help_topic_get_content_casual(self):
        """Test getting casual content."""
        from comfy_headless.help_system import HelpLevel, HelpTopic

        topic = HelpTopic(
            id="test", title="Test", eli5="Simple", casual="Regular", developer="Technical"
        )

        assert topic.get_content(HelpLevel.CASUAL) == "Regular"

    def test_help_topic_get_content_developer(self):
        """Test getting developer content."""
        from comfy_headless.help_system import HelpLevel, HelpTopic

        topic = HelpTopic(
            id="test", title="Test", eli5="Simple", casual="Regular", developer="Technical"
        )

        assert topic.get_content(HelpLevel.DEVELOPER) == "Technical"

    def test_help_topic_format_full(self):
        """Test full formatting."""
        from comfy_headless.help_system import HelpLevel, HelpTopic

        topic = HelpTopic(
            id="test",
            title="Test Topic",
            eli5="Simple",
            casual="Regular",
            developer="Technical",
            see_also=["other_topic"],
        )

        formatted = topic.format_full(HelpLevel.CASUAL)

        assert "# Test Topic" in formatted
        assert "Regular" in formatted
        assert "other_topic" in formatted

    def test_help_topic_format_with_examples(self):
        """Test formatting with examples."""
        from comfy_headless.help_system import HelpLevel, HelpTopic

        topic = HelpTopic(
            id="test",
            title="Test",
            eli5="Simple",
            casual="Regular",
            developer="Technical",
            examples=["example_code()"],
        )

        formatted = topic.format_full(HelpLevel.DEVELOPER)

        assert "## Examples" in formatted
        assert "example_code()" in formatted


class TestGetHelp:
    """Test get_help function."""

    def test_get_help_generation(self):
        """Test getting help for generation topic."""
        from comfy_headless.help_system import get_help

        help_text = get_help("generation")
        assert "Image Generation" in help_text

    def test_get_help_prompts(self):
        """Test getting help for prompts topic."""
        from comfy_headless.help_system import get_help

        help_text = get_help("prompts")
        assert "Prompts" in help_text or "prompt" in help_text.lower()

    def test_get_help_presets(self):
        """Test getting help for presets topic."""
        from comfy_headless.help_system import get_help

        help_text = get_help("presets")
        assert "Presets" in help_text or "preset" in help_text.lower()

    def test_get_help_video(self):
        """Test getting help for video topic."""
        from comfy_headless.help_system import get_help

        help_text = get_help("video")
        assert "Video" in help_text

    def test_get_help_with_level(self):
        """Test getting help at specific level."""
        from comfy_headless.help_system import HelpLevel, get_help

        help_eli5 = get_help("generation", level=HelpLevel.ELI5)
        help_dev = get_help("generation", level=HelpLevel.DEVELOPER)

        # Developer should have more content
        assert len(help_dev) >= len(help_eli5)

    def test_get_help_unknown_topic(self):
        """Test getting help for unknown topic."""
        from comfy_headless.help_system import get_help

        help_text = get_help("nonexistent_topic_xyz")
        assert "not found" in help_text.lower()
        assert "Available topics" in help_text

    def test_get_help_error_format(self):
        """Test getting help for error code."""
        from comfy_headless.help_system import get_help

        help_text = get_help("error:COMFYUI_OFFLINE")
        assert "Offline" in help_text or "offline" in help_text.lower()

    def test_get_help_error_without_prefix(self):
        """Test getting help for error without prefix."""
        from comfy_headless.help_system import get_help

        help_text = get_help("comfyui_offline")
        assert "Offline" in help_text or "not found" in help_text.lower()


class TestGetHelpForError:
    """Test get_help_for_error function."""

    def test_get_help_for_error_offline(self):
        """Test help for offline error."""
        from comfy_headless.help_system import get_help_for_error

        help_text = get_help_for_error("COMFYUI_OFFLINE")
        assert "ComfyUI" in help_text

    def test_get_help_for_error_timeout(self):
        """Test help for timeout error."""
        from comfy_headless.help_system import get_help_for_error

        help_text = get_help_for_error("GENERATION_TIMEOUT")
        assert "timeout" in help_text.lower() or "Timeout" in help_text

    def test_get_help_for_error_vram(self):
        """Test help for VRAM error."""
        from comfy_headless.help_system import get_help_for_error

        help_text = get_help_for_error("INSUFFICIENT_VRAM")
        assert "VRAM" in help_text or "memory" in help_text.lower()


class TestListTopics:
    """Test list_topics function."""

    def test_list_topics_returns_list(self):
        """Test list_topics returns a list."""
        from comfy_headless.help_system import list_topics

        topics = list_topics()
        assert isinstance(topics, list)
        assert len(topics) > 0

    def test_list_topics_contains_core_topics(self):
        """Test list_topics contains core topics."""
        from comfy_headless.help_system import list_topics

        topics = list_topics()
        assert "generation" in topics
        assert "prompts" in topics
        assert "presets" in topics

    def test_list_topics_sorted(self):
        """Test list_topics is sorted."""
        from comfy_headless.help_system import list_topics

        topics = list_topics()
        assert topics == sorted(topics)


class TestSearchHelp:
    """Test search_help function."""

    def test_search_help_finds_topics(self):
        """Test search finds matching topics."""
        from comfy_headless.help_system import search_help

        results = search_help("image")
        assert len(results) > 0

    def test_search_help_case_insensitive(self):
        """Test search is case insensitive."""
        from comfy_headless.help_system import search_help

        results_lower = search_help("image")
        results_upper = search_help("IMAGE")
        assert results_lower == results_upper

    def test_search_help_no_results(self):
        """Test search with no results."""
        from comfy_headless.help_system import search_help

        results = search_help("xyznonexistent123")
        assert len(results) == 0


class TestHelpContext:
    """Test contextual help functions."""

    def test_set_context_operation(self):
        """Test setting context operation."""
        from comfy_headless.help_system import _help_context, set_context

        set_context(operation="generation")
        assert _help_context.current_operation == "generation"

    def test_set_context_error(self):
        """Test setting context error."""
        from comfy_headless.help_system import _help_context, set_context

        set_context(error="COMFYUI_OFFLINE")
        assert _help_context.last_error == "COMFYUI_OFFLINE"

    def test_get_contextual_help_with_error(self):
        """Test contextual help returns error help."""
        from comfy_headless.help_system import _help_context, get_contextual_help, set_context

        # Clear and set context
        _help_context.last_error = None
        _help_context.current_operation = None

        set_context(error="COMFYUI_OFFLINE")
        help_text = get_contextual_help()

        assert "Offline" in help_text or "ComfyUI" in help_text

    def test_get_contextual_help_with_operation(self):
        """Test contextual help returns operation help."""
        from comfy_headless.help_system import _help_context, get_contextual_help, set_context

        # Clear context
        _help_context.last_error = None
        _help_context.current_operation = None

        set_context(operation="video")
        help_text = get_contextual_help()

        assert "Video" in help_text or "video" in help_text.lower()

    def test_get_contextual_help_default(self):
        """Test contextual help default."""
        from comfy_headless.help_system import _help_context, get_contextual_help

        # Clear context
        _help_context.last_error = None
        _help_context.current_operation = None

        help_text = get_contextual_help()
        # Should return generation help by default
        assert "Generation" in help_text or "generation" in help_text.lower()


class TestGetSuggestionsForError:
    """Test get_suggestions_for_error function."""

    def test_get_suggestions_offline(self):
        """Test suggestions for offline error."""
        from comfy_headless.help_system import get_suggestions_for_error

        suggestions = get_suggestions_for_error("COMFYUI_OFFLINE")
        assert isinstance(suggestions, list)

    def test_get_suggestions_unknown_error(self):
        """Test suggestions for unknown error."""
        from comfy_headless.help_system import get_suggestions_for_error

        suggestions = get_suggestions_for_error("UNKNOWN_ERROR_XYZ")
        assert suggestions == []


class TestFormatHelpers:
    """Test format helper functions."""

    def test_format_quick_help(self):
        """Test format_quick_help."""
        from comfy_headless.help_system import format_quick_help

        quick = format_quick_help("generation")
        # Should be short ELI5 content
        assert len(quick) < 200

    def test_format_quick_help_unknown(self):
        """Test format_quick_help for unknown topic."""
        from comfy_headless.help_system import format_quick_help

        quick = format_quick_help("nonexistent")
        assert "No quick help" in quick

    def test_format_help_list(self):
        """Test format_help_list."""
        from comfy_headless.help_system import format_help_list

        help_list = format_help_list()

        assert "# Available Help Topics" in help_list
        assert "## Core" in help_list or "Core" in help_list


class TestBuiltInTopics:
    """Test built-in help topics."""

    def test_generation_topic_exists(self):
        """Test generation topic exists."""
        from comfy_headless.help_system import HELP_TOPICS

        assert "generation" in HELP_TOPICS

    def test_prompts_topic_exists(self):
        """Test prompts topic exists."""
        from comfy_headless.help_system import HELP_TOPICS

        assert "prompts" in HELP_TOPICS

    def test_presets_topic_exists(self):
        """Test presets topic exists."""
        from comfy_headless.help_system import HELP_TOPICS

        assert "presets" in HELP_TOPICS

    def test_video_topic_exists(self):
        """Test video topic exists."""
        from comfy_headless.help_system import HELP_TOPICS

        assert "video" in HELP_TOPICS

    def test_health_checks_topic_exists(self):
        """Test health_checks topic exists."""
        from comfy_headless.help_system import HELP_TOPICS

        assert "health_checks" in HELP_TOPICS

    def test_workflows_topic_exists(self):
        """Test workflows topic exists."""
        from comfy_headless.help_system import HELP_TOPICS

        assert "workflows" in HELP_TOPICS

    def test_error_topics_exist(self):
        """Test error topics exist."""
        from comfy_headless.help_system import HELP_TOPICS

        error_topics = [t for t in HELP_TOPICS if t.startswith("error:")]
        assert len(error_topics) >= 3  # Should have at least 3 error topics

    def test_all_topics_have_content(self):
        """Test all topics have content at all levels."""
        from comfy_headless.help_system import HELP_TOPICS

        for topic_id, topic in HELP_TOPICS.items():
            assert topic.eli5, f"{topic_id} missing eli5"
            assert topic.casual, f"{topic_id} missing casual"
            assert topic.developer, f"{topic_id} missing developer"
