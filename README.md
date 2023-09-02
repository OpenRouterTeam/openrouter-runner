# OpenRouter Runner

## First-time Setup

1. [Install Poetry](https://python-poetry.org/docs/#system-requirements)
2. For Modal runner:

```sh
cd modal
poetry install
poetry shell
modal token new
```

3. To run Scripts:

```sh
pnpm i
pnpm x scripts/${script-name}
```
