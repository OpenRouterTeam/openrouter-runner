# PREREQ:

1.  Create a modal secret group
    `HUGGINGFACE_TOKEN = <your huggingface token>`
    with name "huggingface"
2.  Create a modal secret group
    `tuner_API_KEY = <generate a random key>`
    with name "ext-api-key"
3.  Make sure `pwd` is `/modal`
4.  `modal config set-environment dev # or main`

## Adding new container

1. Copy one of the container file in `tuner/containers`
2. Adapt the class name, image, machine type, engine, etc...
3. Add the container to [](./containers/__init__.py)
4. Add models to be run with the containers in a dedicated list
5. Run: `modal run tuner::download`

## Adding new model

1. Add the model HF ID to [](./containers/__init__.py) under a list
2. Run: `modal run tuner::download`

## Configuration

- For stub config, see: [](./shared/common.py)
- For containers config, see: [](./containers/__init__.py)
- For endpoint configs, see: [](./__init__.py)

## Deploying

Run: `modal deploy tuner`
