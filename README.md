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

## Deploying + Testing

### Modal

```sh
cd modal
poetry shell  # SKIP if already in shell
modal deploy ${filename}

# After deployment is done:
pnpm x scripts/test-modal.ts
```
