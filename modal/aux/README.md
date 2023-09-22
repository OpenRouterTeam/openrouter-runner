# PREREQ:

1.  Create a modal secret group
    `HUGGINGFACE_TOKEN = <your huggingface token>`
    with name "huggingface"
2.  Create a modal secret group
    `AUX_API_KEY = <generate a random key>`
    with name "ext-api-key"
3.  Make sure `pwd` is `/modal`

## Add new engine

1. Copy one of the engine file in `aux/engine`
2. Adapt the class name, image, machine type etc...
3. Add the engine to [](./engines/__init__.py) with a dedicated list of model to run with it
4. Run: `modal run aux/download.py --env=dev # or --env=main`

## Add new model

1. Add the model HF ID to [](./engines/__init__.py), put it under an engine list
2. Runt the download script

## Configuration

- For stub config, see: [](./shared/common.py)
- For engine config, see: [](./engines/__init__.py)
- For endpoint configs, see: [](./main.py)

## Deployment

Run: `modal deploy aux/main.py --env=dev # or --env=main`
