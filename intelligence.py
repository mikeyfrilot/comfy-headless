"""
Comfy Headless - Prompt Intelligence Engine
============================================

AI-powered prompt analysis, enhancement, and workflow recommendation.
Makes image generation accessible by understanding user intent.

Features:
- Intent detection (portrait, landscape, character, etc.)
- Style detection (photorealistic, anime, cinematic, etc.)
- Mood analysis
- Smart prompt enhancement
- Style-specific negative prompt generation
- Workflow recommendation based on analysis

v2.4.0 Enhancements (2026 Best Practices):
- Meta-prompting: Use LLM to improve prompts
- Prompt caching with prefix-based optimization
- Chain-of-thought reasoning for complex prompts
- Few-shot examples for better enhancement
- Prompt versioning and A/B testing support
- Structured input/output (JSON mode)
"""

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from .config import settings
from .logging_config import get_logger
from .retry import get_circuit_breaker

logger = get_logger(__name__)

__all__ = [
    # Main classes
    "PromptIntelligence",
    "PromptAnalysis",
    "EnhancedPrompt",
    # Caching
    "PromptCache",
    "get_prompt_cache",
    # A/B Testing
    "PromptVersion",
    "PromptABTester",
    # Few-shot
    "FEW_SHOT_ENHANCEMENT_EXAMPLES",
    "get_few_shot_prompt",
    "get_few_shot_examples",
    "CHAIN_OF_THOUGHT_TEMPLATE",
    # Convenience functions
    "get_intelligence",
    "analyze_prompt",
    "enhance_prompt",
    "quick_enhance",
    "sanitize_prompt",
]


# =============================================================================
# SECURITY: Prompt Sanitization
# =============================================================================


def sanitize_prompt(prompt: str, max_length: int = 2000) -> str:
    """Sanitize user prompt to prevent injection attacks."""
    if not prompt:
        return ""

    prompt = prompt[:max_length]
    prompt = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", prompt)

    # Filter potential injection patterns
    injection_patterns = [
        r"(?i)ignore\s+(previous|all|above)\s+instructions?",
        r"(?i)system\s*:\s*",
        r"(?i)assistant\s*:\s*",
        r"(?i)user\s*:\s*",
        r"(?i)<\|.*?\|>",
        r"(?i)\[INST\]",
        r"(?i)\[/INST\]",
    ]

    for pattern in injection_patterns:
        prompt = re.sub(pattern, "", prompt)

    return prompt.strip()


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class PromptAnalysis:
    """Result of analyzing a user prompt."""

    original: str
    intent: str = "general"
    subjects: list[str] = field(default_factory=list)
    styles: list[str] = field(default_factory=list)
    mood: str = "neutral"
    complexity: float = 0.5
    suggested_aspect: str = "square"
    suggested_workflow: str = "txt2img_standard"
    suggested_preset: str = "quality"
    confidence: float = 0.5


@dataclass
class EnhancedPrompt:
    """Result of enhancing a prompt."""

    original: str
    enhanced: str
    negative: str
    additions: list[str] = field(default_factory=list)
    reasoning: str = ""
    # v2.4: Added versioning support
    version: str = "1.0.0"
    prompt_hash: str = ""
    created_at: float = field(default_factory=time.time)


# =============================================================================
# PROMPT CACHING (2026 Best Practice: LRU with OrderedDict)
# =============================================================================


class PromptCache:
    """
    Cache for prompt analysis and enhancement results.

    Based on 2026 best practices:
    - LRU eviction using OrderedDict (O(1) instead of O(n))
    - TTL-based expiration
    - Hash-based deduplication
    - Move-to-end on access for true LRU behavior
    """

    def __init__(self, max_size: int = 500, ttl_seconds: int = 600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        # Use OrderedDict for O(1) LRU eviction
        from collections import OrderedDict

        self._analysis_cache: OrderedDict[str, tuple[PromptAnalysis, float]] = OrderedDict()
        self._enhancement_cache: OrderedDict[str, tuple[EnhancedPrompt, float]] = OrderedDict()

    def _hash_prompt(self, prompt: str, style: str = "") -> str:
        """Create cache key from prompt."""
        content = f"{prompt.lower().strip()}:{style}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def get_analysis(self, prompt: str) -> PromptAnalysis | None:
        """Get cached analysis if valid."""
        key = self._hash_prompt(prompt)
        if key not in self._analysis_cache:
            return None

        analysis, timestamp = self._analysis_cache[key]
        if time.time() - timestamp > self.ttl_seconds:
            del self._analysis_cache[key]
            return None

        # Move to end (most recently used)
        self._analysis_cache.move_to_end(key)
        logger.debug(f"Analysis cache hit: {key}")
        return analysis

    def set_analysis(self, prompt: str, analysis: PromptAnalysis):
        """Cache analysis result."""
        key = self._hash_prompt(prompt)
        self._evict_if_full()
        self._analysis_cache[key] = (analysis, time.time())
        self._analysis_cache.move_to_end(key)

    def get_enhancement(self, prompt: str, style: str) -> EnhancedPrompt | None:
        """Get cached enhancement if valid."""
        key = self._hash_prompt(prompt, style)
        if key not in self._enhancement_cache:
            return None

        enhanced, timestamp = self._enhancement_cache[key]
        if time.time() - timestamp > self.ttl_seconds:
            del self._enhancement_cache[key]
            return None

        # Move to end (most recently used)
        self._enhancement_cache.move_to_end(key)
        logger.debug(f"Enhancement cache hit: {key}")
        return enhanced

    def set_enhancement(self, prompt: str, style: str, enhanced: EnhancedPrompt):
        """Cache enhancement result."""
        key = self._hash_prompt(prompt, style)
        self._evict_if_full()
        self._enhancement_cache[key] = (enhanced, time.time())
        self._enhancement_cache.move_to_end(key)

    def _evict_if_full(self):
        """Evict oldest (LRU) entries if cache is full. O(1) operation."""
        total = len(self._analysis_cache) + len(self._enhancement_cache)
        while total >= self.max_size:
            # Evict from whichever cache is larger (pop first = oldest/LRU)
            if len(self._analysis_cache) >= len(self._enhancement_cache):
                if self._analysis_cache:
                    self._analysis_cache.popitem(last=False)  # O(1) pop oldest
            else:
                if self._enhancement_cache:
                    self._enhancement_cache.popitem(last=False)  # O(1) pop oldest
            total = len(self._analysis_cache) + len(self._enhancement_cache)

    def clear(self):
        """Clear all caches."""
        self._analysis_cache.clear()
        self._enhancement_cache.clear()

    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "analysis_entries": len(self._analysis_cache),
            "enhancement_entries": len(self._enhancement_cache),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds,
        }


# Global cache instance
_prompt_cache = PromptCache()


def get_prompt_cache() -> PromptCache:
    """Get the global prompt cache."""
    return _prompt_cache


# =============================================================================
# PROMPT VERSIONING AND A/B TESTING (2026 Best Practice)
# =============================================================================


@dataclass
class PromptVersion:
    """
    A versioned prompt variant for A/B testing.

    Supports comparing different enhancement strategies.
    """

    id: str
    prompt: str
    version: str
    variant: str  # e.g., "A", "B", "control"
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "version": self.version,
            "variant": self.variant,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }


class PromptABTester:
    """
    A/B testing for prompt enhancement strategies.

    Usage:
        tester = PromptABTester()
        tester.register_variant("A", enhance_v1)
        tester.register_variant("B", enhance_v2)
        result = tester.get_variant(prompt)  # Returns "A" or "B"
    """

    def __init__(self, traffic_split: dict[str, float] = None):
        """
        Initialize A/B tester.

        Args:
            traffic_split: e.g., {"A": 0.9, "B": 0.1} for 90/10 split
        """
        self.traffic_split = traffic_split or {"A": 0.5, "B": 0.5}
        self._variants: dict[str, Any] = {}
        self._results: list[dict[str, Any]] = []

    def register_variant(self, name: str, enhancer: Any):
        """Register an enhancement variant."""
        self._variants[name] = enhancer

    def get_variant(self, prompt: str) -> str:
        """
        Get variant for a prompt based on traffic split.

        Uses deterministic hashing for consistency.
        """
        # Hash prompt for deterministic assignment
        h = int(hashlib.md5(prompt.encode()).hexdigest(), 16)
        threshold = h % 100 / 100.0

        cumulative = 0.0
        for variant, weight in self.traffic_split.items():
            cumulative += weight
            if threshold < cumulative:
                return variant

        return list(self.traffic_split.keys())[0]

    def record_result(
        self, prompt: str, variant: str, success: bool, metrics: dict[str, Any] = None
    ):
        """Record A/B test result for analysis."""
        self._results.append(
            {
                "prompt_hash": hashlib.md5(prompt.encode()).hexdigest()[:8],
                "variant": variant,
                "success": success,
                "metrics": metrics or {},
                "timestamp": time.time(),
            }
        )

    def get_results_summary(self) -> dict[str, Any]:
        """Get summary of A/B test results."""
        summary = {}
        for variant in self.traffic_split:
            variant_results = [r for r in self._results if r["variant"] == variant]
            if variant_results:
                success_count = sum(1 for r in variant_results if r["success"])
                summary[variant] = {
                    "total": len(variant_results),
                    "success": success_count,
                    "success_rate": success_count / len(variant_results),
                }
            else:
                summary[variant] = {"total": 0, "success": 0, "success_rate": 0.0}
        return summary


# =============================================================================
# FEW-SHOT EXAMPLES (2026 Best Practice: Context engineering)
# =============================================================================

FEW_SHOT_ENHANCEMENT_EXAMPLES = [
    {
        "input": "a cat",
        "output": "a fluffy orange cat with bright green eyes, sitting gracefully, soft natural lighting, highly detailed fur texture, professional pet photography",
        "style": "detailed",
    },
    {
        "input": "sunset",
        "output": "a breathtaking golden sunset over calm ocean waters, dramatic clouds with orange and purple hues, god rays breaking through, cinematic composition, 8k photography",
        "style": "detailed",
    },
    {
        "input": "portrait of a woman",
        "output": "portrait of an elegant woman, soft rembrandt lighting, detailed skin texture, expressive eyes, shallow depth of field, professional studio photography",
        "style": "balanced",
    },
    {
        "input": "cyberpunk city",
        "output": "neon-lit cyberpunk cityscape at night, rain-slicked streets, holographic advertisements, towering skyscrapers, cinematic atmosphere, blade runner style",
        "style": "creative",
    },
]


def _load_custom_few_shot_examples() -> list[dict[str, str]]:
    """Load custom few-shot examples from config path if specified."""
    from pathlib import Path

    path = settings.ollama.few_shot_examples_path
    if not path:
        return []

    try:
        p = Path(path)
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                examples = json.load(f)
                if isinstance(examples, list):
                    logger.debug(f"Loaded {len(examples)} custom few-shot examples from {path}")
                    return examples
    except Exception as e:
        logger.warning(f"Failed to load custom few-shot examples from {path}: {e}")

    return []


# Cache for custom examples
_custom_few_shot_examples: list[dict[str, str]] | None = None


def get_few_shot_examples() -> list[dict[str, str]]:
    """Get all few-shot examples (custom + built-in)."""
    global _custom_few_shot_examples
    if _custom_few_shot_examples is None:
        _custom_few_shot_examples = _load_custom_few_shot_examples()

    # Custom examples take priority, then built-in
    return _custom_few_shot_examples + FEW_SHOT_ENHANCEMENT_EXAMPLES


def get_few_shot_prompt(style: str = "balanced") -> str:
    """
    Generate few-shot prompt examples for meta-prompting.

    Based on 2026 best practice: provide examples to guide LLM.
    Supports custom examples via COMFY_HEADLESS_OLLAMA__FEW_SHOT_EXAMPLES_PATH.
    """
    all_examples = get_few_shot_examples()
    examples = [e for e in all_examples if e.get("style") == style]
    if not examples:
        examples = all_examples[:2]

    prompt_parts = ["Here are examples of prompt enhancement:\n"]
    for ex in examples:
        prompt_parts.append(f"Input: {ex['input']}")
        prompt_parts.append(f"Output: {ex['output']}\n")

    return "\n".join(prompt_parts)


# =============================================================================
# CHAIN-OF-THOUGHT PROMPTS (2026 Best Practice)
# =============================================================================

CHAIN_OF_THOUGHT_TEMPLATE = """Analyze this image prompt step by step:

Prompt: {prompt}

Step 1 - Identify the main subject:
Step 2 - Detect the intended style (photo, anime, painting, etc.):
Step 3 - Determine the mood/atmosphere:
Step 4 - List missing visual details:
Step 5 - Enhanced prompt:

Output only the enhanced prompt on the last line."""


# =============================================================================
# INTENT KEYWORDS (Comprehensive)
# =============================================================================

INTENT_KEYWORDS = {
    "portrait": [
        "portrait",
        "face",
        "headshot",
        "person",
        "man",
        "woman",
        "girl",
        "boy",
        "character",
        "bust",
        "close-up",
        "closeup",
        "selfie",
        "profile",
        "model",
        "beauty",
        "mugshot",
        "id photo",
        "passport",
        "professional photo",
        "actor",
        "actress",
        "celebrity",
        "influencer",
        "eyes",
        "lips",
    ],
    "landscape": [
        "landscape",
        "scenery",
        "vista",
        "panorama",
        "mountain",
        "forest",
        "ocean",
        "beach",
        "city",
        "cityscape",
        "skyline",
        "nature",
        "outdoor",
        "environment",
        "horizon",
        "valley",
        "desert",
        "jungle",
        "countryside",
        "field",
        "meadow",
        "river",
        "lake",
        "waterfall",
        "sunset",
        "sunrise",
        "night sky",
        "aurora",
    ],
    "character": [
        "character",
        "warrior",
        "knight",
        "wizard",
        "mage",
        "hero",
        "villain",
        "anime",
        "manga",
        "fantasy",
        "rpg",
        "game character",
        "oc",
        "original character",
        "adventurer",
        "paladin",
        "rogue",
        "assassin",
        "samurai",
        "ninja",
        "pirate",
        "soldier",
        "commander",
        "princess",
        "prince",
        "queen",
        "king",
        "elf",
        "dwarf",
        "full body",
        "standing",
        "action pose",
        "dynamic pose",
    ],
    "scene": [
        "scene",
        "action",
        "battle",
        "fight",
        "dancing",
        "walking",
        "sitting",
        "standing",
        "running",
        "flying",
        "dramatic",
        "epic",
        "moment",
        "event",
        "celebration",
        "ceremony",
        "meeting",
        "conversation",
        "chase",
        "escape",
        "exploration",
        "discovery",
        "journey",
        "quest",
        "confrontation",
        "duel",
    ],
    "object": [
        "product",
        "item",
        "object",
        "food",
        "car",
        "vehicle",
        "weapon",
        "sword",
        "still life",
        "macro",
        "detail",
        "jewelry",
        "watch",
        "phone",
        "gadget",
        "furniture",
        "clothing",
        "fashion",
        "accessory",
        "tool",
        "instrument",
        "bottle",
        "glass",
        "cup",
        "plate",
        "book",
        "flower",
        "plant",
        "fruit",
    ],
    "abstract": [
        "abstract",
        "pattern",
        "texture",
        "fractal",
        "geometric",
        "surreal",
        "psychedelic",
        "trippy",
        "dream",
        "concept",
        "symbolic",
        "metaphor",
        "kaleidoscope",
        "mandala",
        "spiral",
        "wave",
        "flow",
        "energy",
        "aura",
        "vortex",
        "portal",
        "dimension",
        "cosmic",
        "infinite",
        "void",
    ],
    "architecture": [
        "building",
        "architecture",
        "interior",
        "room",
        "house",
        "castle",
        "temple",
        "church",
        "modern",
        "ancient",
        "ruins",
        "palace",
        "mansion",
        "skyscraper",
        "tower",
        "bridge",
        "monument",
        "statue",
        "cathedral",
        "mosque",
        "pagoda",
        "fortress",
        "bunker",
        "station",
        "airport",
    ],
    "creature": [
        "creature",
        "monster",
        "beast",
        "animal",
        "dragon",
        "phoenix",
        "unicorn",
        "griffin",
        "hydra",
        "kraken",
        "leviathan",
        "chimera",
        "cerberus",
        "pegasus",
        "dinosaur",
        "alien",
        "mutant",
        "kaiju",
        "titan",
        "colossus",
        "golem",
        "cat",
        "dog",
        "wolf",
        "lion",
        "tiger",
        "bear",
        "bird",
        "eagle",
    ],
}


# =============================================================================
# STYLE KEYWORDS (Comprehensive)
# =============================================================================

STYLE_KEYWORDS = {
    "photorealistic": [
        "photo",
        "photograph",
        "photorealistic",
        "realistic",
        "real",
        "raw",
        "dslr",
        "canon",
        "nikon",
        "sony",
        "35mm",
        "85mm",
        "50mm",
        "camera",
        "lens",
        "f/1.4",
        "f/2.8",
        "bokeh",
        "depth of field",
        "dof",
        "sharp",
        "crisp",
        "natural lighting",
        "studio lighting",
        "flash",
        "softbox",
        "reflector",
    ],
    "anime": [
        "anime",
        "manga",
        "waifu",
        "kawaii",
        "chibi",
        "cel shaded",
        "2d",
        "japanese",
        "light novel",
        "visual novel",
        "ghibli",
        "makoto shinkai",
        "kyoto animation",
        "trigger",
        "ufotable",
        "mappa",
        "wit studio",
        "shonen",
        "shoujo",
        "seinen",
        "isekai",
        "slice of life",
        "moe",
    ],
    "artistic": [
        "painting",
        "oil painting",
        "watercolor",
        "acrylic",
        "impressionist",
        "expressionist",
        "art nouveau",
        "baroque",
        "renaissance",
        "classical",
        "van gogh",
        "monet",
        "rembrandt",
        "picasso",
        "dali",
        "klimt",
        "mucha",
        "gouache",
        "pastel",
        "charcoal",
        "sketch",
        "drawing",
        "illustration",
    ],
    "digital_art": [
        "digital art",
        "cg",
        "3d render",
        "blender",
        "unreal engine",
        "octane",
        "ray tracing",
        "artstation",
        "deviantart",
        "concept art",
        "keyshot",
        "vray",
        "cinema 4d",
        "maya",
        "zbrush",
        "substance painter",
        "houdini",
        "cgi",
        "vfx",
        "matte painting",
        "environment art",
        "character design",
    ],
    "cinematic": [
        "cinematic",
        "movie",
        "film",
        "hollywood",
        "dramatic lighting",
        "volumetric",
        "atmospheric",
        "moody",
        "epic",
        "blockbuster",
        "imax",
        "anamorphic",
        "widescreen",
        "film grain",
        "color grading",
        "lut",
        "cinematography",
        "director",
        "movie still",
        "film still",
    ],
    "fantasy": [
        "fantasy",
        "magical",
        "mystical",
        "ethereal",
        "enchanted",
        "fairy tale",
        "mythology",
        "dragon",
        "elf",
        "dwarf",
        "tolkien",
        "dungeons and dragons",
        "world of warcraft",
        "elder scrolls",
        "dark souls",
        "witcher",
        "lotr",
        "hogwarts",
        "narnia",
        "middle earth",
        "fae",
        "faerie",
        "mythical",
    ],
    "scifi": [
        "sci-fi",
        "scifi",
        "science fiction",
        "futuristic",
        "cyberpunk",
        "steampunk",
        "dieselpunk",
        "solarpunk",
        "biopunk",
        "space",
        "alien",
        "robot",
        "mech",
        "android",
        "cyborg",
        "neon",
        "hologram",
        "laser",
        "star wars",
        "star trek",
        "blade runner",
        "ghost in the shell",
        "akira",
    ],
    "horror": [
        "horror",
        "dark",
        "gothic",
        "creepy",
        "scary",
        "nightmare",
        "grim",
        "sinister",
        "ominous",
        "evil",
        "demon",
        "zombie",
        "ghost",
        "spirit",
        "haunted",
        "cursed",
        "occult",
        "lovecraft",
        "eldritch",
        "cosmic horror",
        "body horror",
        "psychological horror",
        "slasher",
        "gore",
        "blood",
    ],
    "minimalist": [
        "minimalist",
        "minimal",
        "simple",
        "clean",
        "modern",
        "sleek",
        "geometric",
        "flat",
        "negative space",
        "white space",
        "swiss design",
        "bauhaus",
        "scandinavian",
        "japanese",
        "zen",
        "muji",
    ],
}


# =============================================================================
# MOOD KEYWORDS
# =============================================================================

MOOD_KEYWORDS = {
    "dramatic": [
        "dramatic",
        "intense",
        "powerful",
        "epic",
        "grand",
        "majestic",
        "imposing",
        "striking",
    ],
    "peaceful": [
        "peaceful",
        "calm",
        "serene",
        "tranquil",
        "relaxing",
        "gentle",
        "soothing",
        "quiet",
    ],
    "dark": [
        "dark",
        "moody",
        "ominous",
        "sinister",
        "gloomy",
        "mysterious",
        "shadowy",
        "foreboding",
    ],
    "bright": ["bright", "cheerful", "happy", "joyful", "vibrant", "lively", "colorful", "sunny"],
    "romantic": [
        "romantic",
        "love",
        "passionate",
        "intimate",
        "tender",
        "soft",
        "sensual",
        "amorous",
    ],
    "melancholic": [
        "melancholic",
        "sad",
        "lonely",
        "nostalgic",
        "wistful",
        "bittersweet",
        "somber",
    ],
    "energetic": [
        "energetic",
        "dynamic",
        "action",
        "motion",
        "fast",
        "explosive",
        "intense",
        "powerful",
    ],
    "ethereal": [
        "ethereal",
        "dreamy",
        "mystical",
        "otherworldly",
        "magical",
        "surreal",
        "heavenly",
    ],
    "cozy": ["cozy", "warm", "comfortable", "homey", "snug", "inviting", "intimate", "rustic"],
    "epic": [
        "epic",
        "legendary",
        "heroic",
        "monumental",
        "grandiose",
        "spectacular",
        "awe-inspiring",
    ],
}


# =============================================================================
# STYLE-SPECIFIC BOOSTERS
# =============================================================================

STYLE_QUALITY_BOOSTERS = {
    "photorealistic": [
        "8k uhd",
        "hyperrealistic",
        "photorealism",
        "ultra detailed photograph",
        "professional photography",
        "award winning photo",
        "national geographic",
    ],
    "anime": [
        "best quality anime",
        "detailed anime art",
        "beautiful anime style",
        "key visual",
        "official art",
        "clean lineart",
        "vibrant colors",
    ],
    "artistic": [
        "masterpiece painting",
        "museum quality",
        "fine art",
        "exhibition piece",
        "gallery artwork",
        "traditional media",
        "brush strokes visible",
    ],
    "digital_art": [
        "trending on artstation",
        "cgsociety",
        "8k render",
        "unreal engine 5",
        "octane render",
        "ray traced",
        "global illumination",
        "subsurface scattering",
    ],
    "cinematic": [
        "cinematic lighting",
        "film still",
        "movie scene",
        "dramatic composition",
        "volumetric lighting",
        "god rays",
        "lens flare",
        "motion blur",
    ],
    "fantasy": [
        "epic fantasy art",
        "magic atmosphere",
        "enchanted",
        "mythical",
        "legendary",
        "ancient magic",
        "mystical aura",
        "fantasy illustration",
    ],
    "scifi": [
        "sci-fi concept art",
        "futuristic design",
        "advanced technology",
        "cybernetic",
        "neon lights",
        "holographic",
        "chrome",
        "metallic",
    ],
    "horror": [
        "unsettling atmosphere",
        "creepy lighting",
        "dark shadows",
        "ominous presence",
        "eerie fog",
        "nightmare fuel",
        "disturbing",
    ],
}


# =============================================================================
# STYLE-SPECIFIC NEGATIVES
# =============================================================================

STYLE_NEGATIVES = {
    "photorealistic": [
        "cartoon",
        "anime",
        "illustration",
        "painting",
        "drawing",
        "sketch",
        "cgi",
        "3d render",
        "digital art",
        "artistic",
        "stylized",
        "unrealistic",
    ],
    "anime": [
        "realistic",
        "photorealistic",
        "3d",
        "photograph",
        "western",
        "disney",
        "pixar",
        "dreamworks",
        "uncanny valley",
        "hyperrealistic",
    ],
    "artistic": [
        "photorealistic",
        "photograph",
        "3d render",
        "cgi",
        "digital",
        "modern",
        "minimalist",
        "flat design",
        "vector",
        "clipart",
    ],
    "digital_art": [
        "traditional media",
        "oil painting",
        "watercolor",
        "sketch",
        "pencil",
        "charcoal",
        "crayon",
        "pastel",
        "messy",
        "rough",
    ],
    "cinematic": [
        "amateur",
        "home video",
        "low budget",
        "tv show",
        "sitcom",
        "documentary",
        "news footage",
        "security camera",
        "webcam",
    ],
    "fantasy": [
        "modern",
        "contemporary",
        "realistic",
        "mundane",
        "ordinary",
        "scientific",
        "technological",
        "urban",
        "industrial",
    ],
    "scifi": [
        "medieval",
        "ancient",
        "primitive",
        "magical",
        "fantasy",
        "steampunk",
        "low tech",
        "organic",
        "natural",
        "rustic",
    ],
    "horror": [
        "cheerful",
        "happy",
        "bright",
        "colorful",
        "cute",
        "kawaii",
        "wholesome",
        "family friendly",
        "cartoonish",
        "comedic",
    ],
}


# =============================================================================
# BASE NEGATIVE PROMPTS
# =============================================================================

NEGATIVE_DEFAULTS = [
    "ugly",
    "blurry",
    "low quality",
    "distorted",
    "deformed",
    "disfigured",
    "bad anatomy",
    "watermark",
    "signature",
    "text",
    "error",
    "cropped",
    "worst quality",
    "low resolution",
    "jpeg artifacts",
    "duplicate",
    "mutated",
    "out of frame",
    "extra limbs",
    "cloned face",
    "poorly drawn",
    "malformed",
    "missing limbs",
    "floating limbs",
]

QUALITY_BOOSTERS = [
    "masterpiece",
    "best quality",
    "highly detailed",
    "sharp focus",
    "professional",
    "stunning",
    "beautiful",
    "intricate details",
    "ultra detailed",
    "high resolution",
]


# =============================================================================
# PROMPT INTELLIGENCE ENGINE
# =============================================================================


class PromptIntelligence:
    """
    AI-powered prompt analysis and enhancement.

    This is the core of making image generation accessible - it understands
    what users want and helps them express it better.

    v2.4 Enhancements:
    - Caching for analysis and enhancement results
    - Few-shot prompting for better AI enhancement
    - Chain-of-thought reasoning
    - Meta-prompting support
    """

    def __init__(
        self,
        ollama_url: str | None = None,
        model: str | None = None,
        use_cache: bool = True,
        use_few_shot: bool = False,  # v2.5: Default to False for concise output
        verbose: bool = False,  # v2.5: New flag for verbose AI responses
    ):
        self.ollama_url = ollama_url or settings.ollama.url
        self.model = model or settings.ollama.model
        self._client: httpx.Client | None = None
        self._circuit = get_circuit_breaker("ollama")
        # v2.4: Caching and few-shot support
        self.use_cache = use_cache
        self.use_few_shot = use_few_shot
        self.verbose = verbose  # v2.5: Controls prompt length
        self._cache = get_prompt_cache()

        logger.debug(
            "PromptIntelligence initialized",
            extra={"ollama_url": self.ollama_url, "model": self.model},
        )

    def _get_client(self) -> httpx.Client:
        """Get HTTP client with proper timeout configuration."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.Client(
                timeout=httpx.Timeout(
                    connect=settings.ollama.timeout_connect,
                    read=settings.ollama.timeout_enhancement,
                    write=10.0,
                    pool=5.0,
                )
            )
        return self._client

    def _ollama_request_with_retry(
        self, endpoint: str, json_data: dict, timeout: float = None, max_retries: int = 3
    ) -> dict | None:
        """
        Make an Ollama API request with retry logic.

        Uses exponential backoff for transient failures.

        Args:
            endpoint: API endpoint (e.g., "/api/generate")
            json_data: Request body
            timeout: Request timeout (default from settings)
            max_retries: Maximum retry attempts

        Returns:
            Response JSON or None on failure
        """
        from .retry import retry_with_backoff

        url = f"{self.ollama_url}{endpoint}"
        timeout = timeout or settings.ollama.timeout_enhancement

        @retry_with_backoff(
            max_attempts=max_retries,
            backoff_base=1.0,
            backoff_max=10.0,
            exceptions=(httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout),
        )
        def _do_request():
            client = self._get_client()
            response = client.post(url, json=json_data, timeout=timeout)
            response.raise_for_status()
            return response.json()

        try:
            return _do_request()
        except Exception as e:
            logger.warning(f"Ollama request failed after {max_retries} retries: {e}")
            return None

    def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            self._client.close()
            self._client = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def check_ollama(self) -> bool:
        """Check if Ollama is available."""
        try:
            client = self._get_client()
            r = client.get(f"{self.ollama_url}/api/tags", timeout=settings.ollama.timeout_connect)
            return r.status_code == 200
        except Exception as e:
            logger.debug(f"Ollama check failed: {e}")
            return False

    # =========================================================================
    # KEYWORD-BASED ANALYSIS (Fast, no AI required)
    # =========================================================================

    def analyze_keywords(self, prompt: str, skip_cache: bool = False) -> PromptAnalysis:
        """Fast keyword-based analysis without AI."""
        # v2.4: Check cache first
        if self.use_cache and not skip_cache:
            cached = self._cache.get_analysis(prompt)
            if cached:
                logger.debug("Using cached analysis")
                return cached

        prompt_lower = prompt.lower()

        # Detect intent with weighted scoring
        intent_scores = {}
        for intent, keywords in INTENT_KEYWORDS.items():
            score = 0
            for kw in keywords:
                if kw in prompt_lower:
                    score += len(kw.split())  # Multi-word matches score higher
            if score > 0:
                intent_scores[intent] = score

        intent = max(intent_scores, key=intent_scores.get) if intent_scores else "general"

        # Detect styles
        style_scores = {}
        for style, keywords in STYLE_KEYWORDS.items():
            score = 0
            for kw in keywords:
                if kw in prompt_lower:
                    score += len(kw.split())
            if score > 0:
                style_scores[style] = score

        detected_styles = sorted(style_scores.keys(), key=lambda s: style_scores[s], reverse=True)[
            :3
        ]

        # Detect mood
        mood_scores = {}
        for mood, keywords in MOOD_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in prompt_lower)
            if score > 0:
                mood_scores[mood] = score

        mood = max(mood_scores, key=mood_scores.get) if mood_scores else "neutral"

        # Extract subjects
        subjects = self._extract_subjects(prompt)

        # Determine aspect ratio
        if intent in ["portrait", "character"]:
            suggested_aspect = "portrait"
        elif intent in ["landscape", "architecture"]:
            suggested_aspect = "landscape"
        elif intent == "scene":
            suggested_aspect = "widescreen"
        else:
            suggested_aspect = "square"

        # Complexity based on prompt richness
        word_count = len(prompt.split())
        has_quality = any(q in prompt_lower for q in QUALITY_BOOSTERS)
        has_style = len(detected_styles) > 0
        complexity = min(
            1.0, (word_count / 30) * (1.5 if has_quality else 1.0) * (1.2 if has_style else 1.0)
        )

        # Confidence
        total_matches = (
            sum(intent_scores.values()) + sum(style_scores.values()) + sum(mood_scores.values())
        )
        confidence = min(0.85, 0.4 + (total_matches * 0.05))

        # Suggest workflow and preset
        suggested_workflow, suggested_preset = self._suggest_workflow(
            intent, detected_styles, complexity
        )

        analysis = PromptAnalysis(
            original=prompt,
            intent=intent,
            subjects=subjects,
            styles=detected_styles,
            mood=mood,
            complexity=complexity,
            suggested_aspect=suggested_aspect,
            suggested_workflow=suggested_workflow,
            suggested_preset=suggested_preset,
            confidence=confidence,
        )

        # v2.4: Cache the result
        if self.use_cache:
            self._cache.set_analysis(prompt, analysis)

        return analysis

    def _extract_subjects(self, prompt: str) -> list[str]:
        """Extract likely subjects from prompt."""
        subjects = []
        skip_words = {
            "a",
            "an",
            "the",
            "with",
            "and",
            "or",
            "in",
            "on",
            "at",
            "by",
            "highly",
            "detailed",
            "quality",
            "best",
            "beautiful",
            "stunning",
            "masterpiece",
            "realistic",
            "style",
            "art",
            "artwork",
            "of",
            "is",
        }

        words = re.findall(r"\b[a-zA-Z]+\b", prompt.lower())
        seen = set()

        for word in words:
            if word not in skip_words and len(word) > 2 and word not in seen:
                for intent_keywords in INTENT_KEYWORDS.values():
                    if word in intent_keywords:
                        subjects.append(word)
                        seen.add(word)
                        break

        return subjects[:5]

    def _suggest_workflow(
        self, intent: str, styles: list[str], complexity: float
    ) -> tuple[str, str]:
        """Suggest workflow and preset based on analysis."""
        workflow = "txt2img_standard"
        preset = "quality"

        # High complexity or quality-focused styles -> hi-res
        if complexity > 0.7 or "digital_art" in styles or "cinematic" in styles:
            workflow = "txt2img_hires"

        # Intent-based preset selection
        if intent == "portrait":
            preset = "portrait"
        elif intent == "landscape":
            preset = "landscape"
        elif intent in ["scene", "architecture"]:
            preset = "cinematic" if "cinematic" in styles else "landscape"

        # Style-based adjustments
        if "anime" in styles:
            preset = "quality"  # Anime doesn't need hi-res fix usually
            workflow = "txt2img_standard"

        return workflow, preset

    # =========================================================================
    # PROMPT ENHANCEMENT
    # =========================================================================

    def enhance(
        self,
        prompt: str,
        style: str = "balanced",
        analysis: PromptAnalysis | None = None,
        skip_cache: bool = False,
    ) -> EnhancedPrompt:
        """
        Enhance a prompt with quality terms and style-specific additions.

        Styles:
        - minimal: Add only 2-3 essential terms
        - balanced: Add 4-6 specific details
        - detailed: Rich enhancement with many details
        - creative: Artistic interpretation
        """
        # v2.4: Check cache first
        if self.use_cache and not skip_cache:
            cached = self._cache.get_enhancement(prompt, style)
            if cached:
                logger.debug("Using cached enhancement")
                return cached

        if analysis is None:
            analysis = self.analyze_keywords(prompt)

        enhanced = self._enhance_prompt(prompt, style, analysis)
        negative = self._generate_negative(prompt, analysis)
        additions = self._diff_prompts(prompt, enhanced)

        # v2.4: Add hash for versioning
        prompt_hash = hashlib.md5(f"{prompt}:{style}".encode()).hexdigest()[:12]

        result = EnhancedPrompt(
            original=prompt,
            enhanced=enhanced,
            negative=negative,
            additions=additions,
            reasoning=f"Enhanced with {style} style for {analysis.intent} intent",
            prompt_hash=prompt_hash,
        )

        # v2.4: Cache the result
        if self.use_cache:
            self._cache.set_enhancement(prompt, style, result)

        return result

    def _enhance_prompt(self, prompt: str, style: str, analysis: PromptAnalysis) -> str:
        """Enhance the prompt based on style preference."""

        # Base style additions
        style_additions = {
            "minimal": [],
            "balanced": ["highly detailed", "professional quality"],
            "detailed": [
                "masterpiece",
                "best quality",
                "highly detailed",
                "intricate details",
                "sharp focus",
                "8k",
            ],
            "creative": [
                "artistic",
                "creative composition",
                "beautiful lighting",
                "aesthetic",
                "expressive",
            ],
        }

        additions = list(style_additions.get(style, style_additions["balanced"]))

        # Add style-specific quality boosters
        primary_style = analysis.styles[0] if analysis.styles else None
        if primary_style and primary_style in STYLE_QUALITY_BOOSTERS:
            additions.extend(STYLE_QUALITY_BOOSTERS[primary_style][:3])

        # Intent-specific additions
        intent_additions = {
            "portrait": ["beautiful lighting", "detailed face", "expressive eyes", "skin texture"],
            "landscape": [
                "scenic vista",
                "atmospheric perspective",
                "natural lighting",
                "majestic scale",
            ],
            "character": ["full body shot", "dynamic pose", "detailed costume", "character design"],
            "scene": [
                "cinematic composition",
                "dramatic lighting",
                "storytelling",
                "environmental detail",
            ],
            "object": [
                "studio lighting",
                "product photography",
                "clean background",
                "sharp details",
            ],
            "architecture": ["architectural photography", "symmetrical composition", "grand scale"],
            "creature": [
                "creature design",
                "anatomical detail",
                "texture detail",
                "menacing presence",
            ],
            "abstract": [
                "artistic interpretation",
                "color harmony",
                "visual flow",
                "conceptual depth",
            ],
        }

        if analysis.intent in intent_additions:
            additions.extend(intent_additions[analysis.intent][:3])

        # Mood-specific additions
        mood_additions = {
            "dramatic": ["dramatic lighting", "cinematic", "high contrast"],
            "peaceful": ["soft lighting", "serene atmosphere", "gentle tones"],
            "dark": ["moody lighting", "dark atmosphere", "shadows"],
            "bright": ["bright colors", "vibrant", "high key lighting"],
            "ethereal": ["ethereal glow", "dreamy atmosphere", "soft focus"],
            "epic": ["epic scale", "grandiose", "awe-inspiring"],
            "cozy": ["warm lighting", "intimate atmosphere", "golden hour"],
        }

        if analysis.mood in mood_additions:
            additions.extend(mood_additions[analysis.mood][:2])

        # Filter duplicates and existing terms
        prompt_lower = prompt.lower()
        unique_additions = []
        seen = set()
        for a in additions:
            a_lower = a.lower()
            if a_lower not in prompt_lower and a_lower not in seen:
                unique_additions.append(a)
                seen.add(a_lower)

        enhanced = f"{prompt}, {', '.join(unique_additions[:8])}" if unique_additions else prompt

        return enhanced

    def _generate_negative(self, prompt: str, analysis: PromptAnalysis) -> str:
        """Generate an appropriate negative prompt."""
        negatives = list(NEGATIVE_DEFAULTS)

        # Add style-specific negatives
        primary_style = analysis.styles[0] if analysis.styles else None
        if primary_style and primary_style in STYLE_NEGATIVES:
            negatives.extend(STYLE_NEGATIVES[primary_style])

        # Intent-specific negatives
        intent_negatives = {
            "portrait": [
                "bad hands",
                "missing fingers",
                "extra fingers",
                "bad face",
                "asymmetrical eyes",
            ],
            "landscape": ["people", "person", "human", "text", "words", "letters", "watermark"],
            "character": ["bad anatomy", "wrong proportions", "missing limbs", "extra limbs"],
            "object": ["background clutter", "distracting elements", "text", "watermark"],
            "architecture": ["distorted perspective", "warped lines", "floating elements"],
        }

        if analysis.intent in intent_negatives:
            negatives.extend(intent_negatives[analysis.intent])

        # Remove duplicates
        seen = set()
        unique_negatives = []
        for n in negatives:
            if n not in seen:
                seen.add(n)
                unique_negatives.append(n)

        return ", ".join(unique_negatives[:25])

    def _diff_prompts(self, original: str, enhanced: str) -> list[str]:
        """Find what was added during enhancement."""
        original_terms = {t.strip().lower() for t in original.split(",")}
        enhanced_terms = {t.strip().lower() for t in enhanced.split(",")}
        additions = enhanced_terms - original_terms
        return [a for a in additions if a]

    # =========================================================================
    # AI-POWERED ENHANCEMENT (Optional, requires Ollama)
    # =========================================================================

    def enhance_with_ai(
        self,
        prompt: str,
        enhancement_style: str = "balanced",
        use_few_shot: bool = None,
        use_chain_of_thought: bool = False,
    ) -> tuple[str, str, str]:
        """
        AI-powered prompt enhancement using Ollama.

        v2.4 Enhancements:
        - Few-shot examples for better results
        - Chain-of-thought reasoning option
        - Meta-prompting support

        Args:
            prompt: The prompt to enhance
            enhancement_style: "minimal", "balanced", "detailed", or "creative"
            use_few_shot: Include few-shot examples (default: self.use_few_shot)
            use_chain_of_thought: Use step-by-step reasoning

        Returns: (enhanced_prompt, negative_prompt, info)
        """
        if not prompt.strip():
            return "", ", ".join(NEGATIVE_DEFAULTS), "Enter a prompt first"

        if not self.check_ollama():
            # Fall back to keyword-based enhancement
            enhanced = self.enhance(prompt, enhancement_style)
            return enhanced.enhanced, enhanced.negative, "Ollama offline - used keyword enhancement"

        safe_prompt = sanitize_prompt(prompt)

        # Determine if we should use few-shot
        should_use_few_shot = use_few_shot if use_few_shot is not None else self.use_few_shot

        # v2.5: Concise style instructions (match workflows UI behavior)
        style_instructions = {
            "minimal": "Add 2-3 quality terms only.",
            "balanced": "Add lighting, composition, quality details. Stay concise.",
            "detailed": "Add textures, lighting, atmosphere. Under 150 chars added.",
            "creative": "Artistic interpretation with mood and style.",
        }

        instruction = style_instructions.get(enhancement_style, style_instructions["balanced"])

        # v2.5: Simplified system prompt for concise output
        if self.verbose and should_use_few_shot:
            # Verbose mode: include few-shot examples
            system_parts = [f"You enhance image generation prompts. {instruction}"]
            system_parts.append("\n" + get_few_shot_prompt(enhancement_style))
            system_parts.append("Output ONLY the enhanced prompt. Stay under 300 characters.")
            system_prompt = "\n".join(system_parts)
        else:
            # Concise mode (default): minimal system prompt
            system_prompt = f"""Enhance this image prompt. {instruction}
Output ONLY the enhanced prompt, nothing else. Keep under 300 total characters."""

        # v2.4: Use chain-of-thought if requested
        if use_chain_of_thought:
            user_prompt = CHAIN_OF_THOUGHT_TEMPLATE.format(prompt=safe_prompt)
        else:
            user_prompt = f"Enhance this prompt:\n{safe_prompt}"

        # Use retry-enabled Ollama request
        response_data = self._ollama_request_with_retry(
            endpoint="/api/generate",
            json_data={
                "model": self.model,
                "prompt": user_prompt,
                "system": system_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 300 if use_chain_of_thought else 200,
                },
            },
            timeout=settings.ollama.timeout_enhancement,
            max_retries=settings.retry.max_retries,
        )

        if response_data:
            enhanced = response_data.get("response", "").strip()

            # v2.4: For chain-of-thought, extract final line
            if use_chain_of_thought and "\n" in enhanced:
                enhanced = enhanced.strip().split("\n")[-1]

            enhanced = enhanced.strip("\"'")

            # v2.5: Enforce character limit (300 chars max)
            if len(enhanced) > 300:
                # Truncate at last comma before limit to keep clean formatting
                truncated = enhanced[:300]
                last_comma = truncated.rfind(",")
                if last_comma > len(prompt):  # Keep at least original
                    enhanced = truncated[:last_comma]
                else:
                    enhanced = truncated
                logger.debug(f"Truncated enhanced prompt to {len(enhanced)} chars")

            # Generate smart negative
            analysis = self.analyze_keywords(prompt)
            negative = self._generate_negative(prompt, analysis)

            added_words = len(enhanced.split()) - len(prompt.split())
            mode = (
                "CoT" if use_chain_of_thought else ("few-shot" if should_use_few_shot else "basic")
            )
            info = f"Added ~{added_words} words | Style: {enhancement_style} | Mode: {mode}"

            logger.debug(
                "AI enhancement complete",
                extra={"style": enhancement_style, "added_words": added_words, "mode": mode},
            )

            return enhanced, negative, info

        # Fall back to keyword enhancement if Ollama failed
        logger.warning("AI enhancement failed after retries, using keyword fallback")
        enhanced = self.enhance(prompt, enhancement_style)
        return enhanced.enhanced, enhanced.negative, "AI unavailable, used keyword enhancement"

    def analyze_with_ai(self, prompt: str) -> str:
        """Quick AI analysis of the prompt with retry support."""
        if not prompt.strip():
            return "Enter a prompt..."

        if not self.check_ollama():
            # Fall back to keyword analysis
            analysis = self.analyze_keywords(prompt)
            return f"**Intent**: {analysis.intent}\n**Styles**: {', '.join(analysis.styles) or 'general'}\n**Mood**: {analysis.mood}\n**Suggested**: {analysis.suggested_preset} preset"

        safe_prompt = sanitize_prompt(prompt)

        # Use retry-enabled request
        response_data = self._ollama_request_with_retry(
            endpoint="/api/generate",
            json_data={
                "model": self.model,
                "prompt": f"In exactly 2 lines, analyze this image prompt. Line 1: detected style/genre. Line 2: suggested improvements (brief).\n\nPrompt: {safe_prompt}",
                "system": "You analyze image generation prompts. Be extremely concise - max 2 short lines.",
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 80},
            },
            timeout=settings.ollama.timeout_analysis,
            max_retries=2,  # Faster timeout for analysis
        )

        if response_data:
            result = response_data.get("response", "").strip()
            logger.debug("AI analysis complete")
            return result

        # Fallback to keyword analysis
        analysis = self.analyze_keywords(prompt)
        return f"**Intent**: {analysis.intent}\n**Styles**: {', '.join(analysis.styles) or 'general'}\n**Mood**: {analysis.mood}\n**Suggested**: {analysis.suggested_preset} preset"

    def generate_variations(self, prompt: str, count: int = 4) -> list[str]:
        """Generate prompt variations with retry support."""
        fallback_variations = [
            f"{prompt}, golden hour lighting",
            f"{prompt}, dramatic atmosphere",
            f"{prompt}, close-up view",
            f"{prompt}, wide angle shot",
        ][:count]

        if not self.check_ollama():
            logger.debug("Using fallback variations (Ollama offline)")
            return fallback_variations

        safe_prompt = sanitize_prompt(prompt)

        # Use retry-enabled request
        response_data = self._ollama_request_with_retry(
            endpoint="/api/generate",
            json_data={
                "model": self.model,
                "prompt": f"Generate {count} variations of this image prompt. Vary: perspective, lighting, mood. One per line, numbered.\n\nOriginal: {safe_prompt}",
                "system": "Generate prompt variations. Output numbered list only, no explanations.",
                "stream": False,
                "options": {"temperature": 0.8, "num_predict": 400},
            },
            timeout=settings.ollama.timeout_enhancement,
            max_retries=2,
        )

        if response_data:
            text = response_data.get("response", "")
            variations = []
            for line in text.split("\n"):
                line = re.sub(r"^\d+[\.\)]\s*", "", line.strip())
                if line and len(line) > 10:
                    variations.append(line)
            logger.debug(f"Generated {len(variations)} variations")
            return variations[:count] if variations else fallback_variations

        return fallback_variations


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_intelligence: PromptIntelligence | None = None


def get_intelligence() -> PromptIntelligence:
    """Get the singleton prompt intelligence instance."""
    global _intelligence
    if _intelligence is None:
        _intelligence = PromptIntelligence()
    return _intelligence


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def analyze_prompt(prompt: str) -> PromptAnalysis:
    """Analyze a prompt and return recommendations."""
    intel = get_intelligence()
    return intel.analyze_keywords(prompt)


def enhance_prompt(prompt: str, style: str = "balanced") -> EnhancedPrompt:
    """Enhance a prompt with quality terms."""
    intel = get_intelligence()
    return intel.enhance(prompt, style)


def quick_enhance(prompt: str, style: str = "balanced") -> tuple[str, str]:
    """Quick enhancement returning (enhanced, negative)."""
    intel = get_intelligence()
    result = intel.enhance(prompt, style)
    return result.enhanced, result.negative
