"""CLI entry point for unbias-plus."""

import argparse
import sys

from unbias_plus.api import serve
from unbias_plus.model import DEFAULT_MODEL
from unbias_plus.pipeline import UnBiasPlus


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.

    """
    parser = argparse.ArgumentParser(
        prog="unbias-plus",
        description="Detect and debias text using a single LLM.",
    )

    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--text",
        type=str,
        help="Text string to analyze.",
    )
    input_group.add_argument(
        "--file",
        type=str,
        help="Path to a .txt file to analyze.",
    )
    input_group.add_argument(
        "--serve",
        action="store_true",
        default=False,
        help="Start the FastAPI server.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"HuggingFace model ID or local path. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        default=False,
        help="Load model in 4-bit quantization to reduce VRAM usage.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output result as raw JSON instead of formatted CLI display.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens to generate. Default: 1024",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host for the API server. Default: 0.0.0.0",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for the API server. Default: 8000",
    )

    return parser.parse_args()


def main() -> None:
    """Run the unbias-plus CLI.

    Examples
    --------
    $ unbias-plus --text "Women are too emotional to lead."
    $ unbias-plus --file article.txt --json
    $ unbias-plus --serve --model path/to/model --port 8000
    $ unbias-plus --serve --load-in-4bit

    """
    args = parse_args()

    if args.serve:
        serve(
            model_name_or_path=args.model,
            host=args.host,
            port=args.port,
            load_in_4bit=args.load_in_4bit,
        )
        return

    if not args.text and not args.file:
        print(
            "Error: one of --text, --file, or --serve is required.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.file:
        try:
            with open(args.file) as f:
                text = f.read()
        except FileNotFoundError:
            print(f"Error: file '{args.file}' not found.", file=sys.stderr)
            sys.exit(1)
    else:
        text = args.text

    pipe = UnBiasPlus(
        model_name_or_path=args.model,
        load_in_4bit=args.load_in_4bit,
        max_new_tokens=args.max_new_tokens,
    )

    if args.json:
        print(pipe.analyze_to_json(text))
    else:
        print(pipe.analyze_to_cli(text))


if __name__ == "__main__":
    main()
