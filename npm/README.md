# evalproof for npm

This package runs [evalproof](https://basantbhattarai.com.np/evalproof/), the LLM eval and CI
regression gate, from `npx`. It is a launcher. evalproof is a Python program published on
[PyPI](https://pypi.org/project/evalproof/), and this package runs the PyPI release that has the
same version number.

```bash
npx evalproof init                 # write a starter suite and a CI workflow
npx evalproof compare base.json candidate.json --gate
```

## What it needs

One of these, tried in this order:

1. [uv](https://docs.astral.sh/uv/getting-started/installation/): runs `uvx --from evalproof==VERSION evalproof ...`.
2. [pipx](https://pipx.pypa.io/): runs `pipx run --spec evalproof==VERSION evalproof ...`.
3. An `evalproof` on your `PATH` that reports exactly the same version, from `pip install evalproof==VERSION`.

If none of them is found, the launcher prints how to install uv and exits with 127. Set
`EVALPROOF_LAUNCHER` to `uvx`, `pipx` or `path` to use only one of them.

Node 18 or newer. The launcher has no dependencies, passes its arguments and its exit code
through, and forwards `SIGINT` and `SIGTERM` to evalproof on Linux and macOS.

## More

The documentation is at <https://basantbhattarai.com.np/evalproof/>. The source is in the
[`npm` directory](https://github.com/technobasant/evalproof/tree/main/npm) of the repository.
Licensed under Apache-2.0.
