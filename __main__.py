"""
Comfy Headless - CLI Entry Point
Run with: python -m comfy_headless
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Comfy Headless - Headless ComfyUI Interface")
    parser.add_argument(
        "--port", "-p", type=int, default=7861, help="Port to run the UI on (default: 7861)"
    )
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8188",
        help="ComfyUI server URL (default: http://localhost:8188)",
    )
    parser.add_argument("--version", "-v", action="store_true", help="Show version and exit")
    parser.add_argument("--check", action="store_true", help="Check feature availability and exit")

    args = parser.parse_args()

    # Version check
    if args.version:
        from . import __version__

        print(f"comfy-headless v{__version__}")
        sys.exit(0)

    # Feature check
    if args.check:
        from . import FEATURES, list_available_features, list_missing_features

        print("Comfy Headless - Feature Check")
        print("=" * 40)
        print("\nInstalled features:")
        for name, desc in list_available_features().items():
            print(f"  [+] {name}: {desc}")
        missing = list_missing_features()
        if missing:
            print("\nMissing features:")
            for name, hint in missing.items():
                print(f"  [ ] {name}: {hint}")
        sys.exit(0)

    # Check if UI extra is installed
    from .feature_flags import FEATURES, get_install_hint

    if not FEATURES.get("ui", False):
        print("Error: UI feature not installed.")
        print(f"Install with: {get_install_hint('ui')}")
        print("\nAlternatively, use comfy-headless as a library:")
        print("  from comfy_headless import ComfyClient")
        print("  client = ComfyClient()")
        print("  result = client.generate_image('your prompt')")
        sys.exit(1)

    # Import launch only after checking feature
    from . import launch

    # Update client URL if specified
    if args.url != "http://localhost:8188":
        from .config import settings

        settings.comfyui.url = args.url.rstrip("/")

    print(f"Starting Comfy Headless on port {args.port}...")
    print(f"ComfyUI URL: {args.url}")

    launch(port=args.port, share=args.share)


if __name__ == "__main__":
    main()
