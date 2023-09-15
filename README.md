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

3. To run a script:

```sh
pnpm i
pnpm x scripts/${script-name}
```

## Deploying + Testing

### Modal

```sh
cd modal
poetry shell  # SKIP if already in shell

modal deploy vllm_runner/${filename} --env main # For production
modal deploy vllm_runner/${filename} --env dev # For dev environment

# After deployment is done:
pnpm x scripts/test-${testname}.ts # For dev environment
pnpm x scripts/test-${testname}.ts prod # For prod environment
```
