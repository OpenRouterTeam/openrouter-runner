# Contributing to OpenRouter Runner

Thank you for your interest in contributing to OpenRouter Runner! We welcome all contributions, big or small. Please read this document to learn how to get started. If you have any questions, please feel free to reach out to us on [Discord](https://discord.gg/tnPTxcYmGf).

## Table of Contents

- [Pre-requisites](#pre-requisites)
- [First-time Setup](#first-time-setup)
- [Workflows](#workflows)
  - [Setup Modal Secrets and Environment](#setup-modal-secrets-and-environment)
  - [Adding a new open-source model](#adding-a-new-open-source-model)
  - [Adding new engine](#adding-new-engine)
  - [Adding new container](#adding-new-container)
- [Deploying + Testing](#deploying--testing)
  - [Modal](#modal)

## Pre-requisites

- Install [pnpm](https://pnpm.io/installation)
- Install [Poetry](https://python-poetry.org/docs/#system-requirements)
- Create a [Modal](https://modal.com/docs/guide) account
- (Optional) Create a GCP Account

## First-time Setup

1. Setup Poetry:

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

Setup lightweight pre-commit hooks for linting:
```sh
pre-commit install
```

2. To run a script:

```sh
pnpm i
pnpm x scripts/${script-name}
```

## Workflows

### Adding a new open-source model

If the model is supported by an existing engine (such as vLLM), and can be run by an existing container, add its HF ID into an existing container group list in the [containers init](../modal/runner/containers/__init__.py).

Else:

1. [Add a new engine](#adding-new-engine)
2. [Add a new container](#adding-new-container)
3. Add the model ID into a new list in the [containers init](../modal/runner/containers/__init__.py).

### Adding new engine

1. Copy one of the engine declaration in [`runner/engines`](../modal/runner/engines)
2. Adapt the class name, init logics, and implement the generation logic

### Adding new container

1. Copy one of the container file in [`runner/containers`](../modal/runner/containers)
2. Adapt the class name, image, machine type, engine, etc...
3. Add the container to the [containers init](../modal/runner/containers/__init__.py)

## Deploying + Testing

Follow the steps in the [Runner README](../modal/runner/README.md). Make sure you test
on your own Modal account before submitting a PR. New Modal accounts should receive $30
in free credits for development.
