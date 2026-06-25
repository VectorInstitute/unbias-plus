# Quickstart

Try UnBias-Plus in about 30 seconds.

=== "pip"

    ```bash
    pip install unbias-plus
    unbias-plus --text "Women are too emotional to lead."
    ```

=== "uv"

    ```bash
    uv sync
    source .venv/bin/activate
    unbias-plus --text "Women are too emotional to lead."
    ```

=== "API + demo UI"

    ```bash
    pip install unbias-plus
    unbias-plus --serve
    # open http://localhost:8000
    ```

=== "Python"

    ```python
    from unbias_plus import UnBiasPlus

    pipe = UnBiasPlus()
    result = pipe.analyze("Women are too emotional to lead.")
    print(result.binary_label)
    print(result.unbiased_text)
    ```

For setup details, deployment patterns, and advanced usage, see [Installation](guides/install.md) and the other [Guides](guides/install.md).
