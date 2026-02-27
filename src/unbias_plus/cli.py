"""CLI entry point for unbias-plus."""

import argparse

from unbias_plus.pipeline import UnBiasPlus


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments including text and model path.

    """
    pass


def main() -> None:
    """Run the unbias-plus CLI.

    Examples
    --------
    $ unbias-plus --text "Women are too emotional to lead."
    $ unbias-plus --file article.txt
    $ unbias-plus --text "..." --model path/to/model --json

    """
    pass


if __name__ == "__main__":
    main()
