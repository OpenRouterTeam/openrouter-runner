# PREREQ:

1.  Create a modal secret group
    `HUGGINGFACE_TOKEN = <your huggingface token>`
    with name "huggingface"
2.  Create a modal secret group
    `RUNNER_API_KEY = <generate a random key>`
    with name "ext-api-key"
3.  Make sure your current directory, `pwd`, is `/modal`
4.  `modal config set-environment dev # or main`

## Adding new container

1. Copy one of the container file in `runner/containers`
2. Adapt the class name, image, machine type, engine, etc...
3. Add the container to [](./containers/__init__.py)
4. Add models to be run with the containers in a dedicated list
5. Run: `modal run runner::download`

## Adding new model

1. Add the model HF ID to [](./containers/__init__.py) under a list
2. Run: `modal run runner::download`

## Configuration

- For stub config, see: [](./shared/common.py)
- For containers config, see: [](./containers/__init__.py)
- For endpoint configs, see: [](./__init__.py)

## Testing a model

1. `cd ..`
2. Load your environment, e.g. `source .env.dev`
3. Run: `pnpm x scripts/test-simple.ts YourModel/Identifier`

Other tests are available in `scripts`:

```shell
pnpm x scripts/test-${testname}.ts Gryphe/MythoMax-L2-13b # For dev environment
pnpm x scripts/test-${testname}.ts Gryphe/MythoMax-L2-13b prod # For prod environment
```

## Deploying

Run: `modal deploy runner`
