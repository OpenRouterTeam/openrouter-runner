# OpenRouter Runner

## First-time Setup

1. [Install Poetry](https://python-poetry.org/docs/#system-requirements)
2. Setup Poetry:

```sh
poetry install
poetry shell
modal token new
```

For intellisense, it's recommended to run vscode via the poetry shell:

```sh
poetry shell
code .
```

3. To run Scripts:

```sh
pnpm i
pnpm x scripts/${script-name}
```

## Deploying + Testing

### Modal

```sh
cd modal
poetry shell  # SKIP if already in shell
modal deploy modal/${filename}

# After deployment is done:
pnpm x scripts/test-modal.ts
```
