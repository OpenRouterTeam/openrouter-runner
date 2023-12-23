# Contributing to OpenRouter Runner

Thank you for your interest in contributing to OpenRouter Runner! We welcome all contributions, big or small. This guide will provide you with detailed steps for setting up your environment, adding new models, deploying, and testing. If you have any questions, please feel free to reach out to us on [Discord](https://discord.gg/tnPTxcYmGf).

## Table of Contents

- [Pre-requisites](#pre-requisites)
- [First-time Setup](#first-time-setup)
  - [Environment and Secrets Setup](#environment-and-secrets-setup)
- [Workflows](#workflows)
  - [Adding a New Open-Source Model](#adding-a-new-open-source-model)
  - [Adding a New Engine](#adding-a-new-engine)
  - [Adding a New Container](#adding-a-new-container)
- [Configuration and Testing](#configuration-and-testing)
- [Deploying](#deploying)
- [Contributions](#contributions)

## Pre-requisites

- Install [pnpm](https://pnpm.io/installation)
- Install [Poetry](https://python-poetry.org/docs/#system-requirements)
- Create a [Modal](https://modal.com/docs/guide) account
- (Optional) Create a GCP Account
- Ensure you have a [Hugging Face Account](https://huggingface.co/)

## First-time Setup

Setting up your development environment is the first step to contributing. Here's how you can get started:

1. **Install Poetry**: Poetry is a tool for dependency management and packaging in Python. Install it by following the instructions on the [official Poetry installation page](https://python-poetry.org/docs/#installation).

2. **Install pnpm**: pnpm is a fast, disk space-efficient package manager. To install it, follow the instructions on the [official pnpm installation page](https://pnpm.io/installation).

3. **Setup Modal CLI**: The Modal CLI is essential for interacting with the Modal platform. Install it using the following command:
    ```shell
    pip install modal
    ```

4. **Run Scripts with pnpm**: Once you have pnpm installed, you can run scripts as follows:
    ```shell
    pnpm run script-name
    ```
    Replace `script-name` with the name of the script you want to execute.

### Environment and Secrets Setup

Properly setting up your environment and secrets is crucial for a secure and efficient development workflow. Here's how to do it:

1. **Set Your Current Directory**: Ensure that your terminal's current working directory is the `/modal` directory within the project. This is where you'll be running most of your commands. Change it using the `cd` command:
    ```shell
    cd path/to/modal
    ```

2. **Create a Development Environment in Modal**: A separate development environment in Modal allows you to test and develop without affecting the production. Set it up using:
    ```shell
    modal environment create dev
    ```
    This command creates a new environment named 'dev'.

3. **Configure Modal CLI**: Configure the Modal CLI to use your newly created development environment:
    ```shell
    modal config set-environment dev
    ```

4. **Create Modal Secrets**:
    - **Hugging Face Token**: To update models you'll need to have a Hugging Face Token and store it. You'd run this script to store the secret in your Modal account.
        ```shell
        modal secret create huggingface HUGGINGFACE_TOKEN=<your huggingface token>
        ```
    - **Runner API Key**: You'll also need to create a unique runner API key, create one and then add it to Modal as follows:
        ```shell
        modal secret create runner-api RUNNER_API_KEY=<generate a strong key>
        ```
    Replace `<your huggingface token>` and `<generate a strong key>` with your actual tokens.

5. **Verify Secrets**: After setting up your secrets, verify that they are correctly stored in your Modal dashboard under the secrets tab.

>[!IMPORTANT]
> Keep your API keys and secrets secure. Do not share them in public repositories or with unauthorized persons.

## Workflows

### Adding a new open-source model

When adding a new model to OpenRouter Runner, you'll be integrating external AI models to enhance the capabilities of the system. Here's how to do it step by step:

1. **Identify the Model**: Determine if the open-source model you wish to add is supported by an existing engine and can run in an existing container. Find the model's Hugging Face ID or equivalent identifier.

2. **Update the Container List**:
    - If the model is compatible with existing infrastructure, simply add its identifier to the relevant list in `runner/containers/__init__.py`.

    ```python
    existing_model_ids = [
        ...,
        "new-model-id",  # Add your new model ID here.
    ]
    ```

3. **Handle Unsupported Models**:
    - If the model isn't supported by existing engines or containers, you'll need to [add a new engine](#adding-a-new-engine) and [add a new container](#adding-a-new-container).

>[!IMPORTANT]
> Always verify the model's license and ensure it's compatible with OpenRouter Runner's usage.

### Adding a new engine

Creating a new engine involves setting up the logic to interact with different types of AI models or adapting existing models to new requirements. Here's a detailed guide based on the structure of engines like the one you provided:

1. **Understand the Existing Engine Structure**:
    - Familiarize yourself with the structure and components of an existing engine, like the `vllm.py`. Notice how it uses parameters (`VllmParams`), handles generation with `generate`, and manages asynchronous operations.

2. **Copy an Existing Engine as a Template**:
    - Use an existing engine file as a starting point. Copy it and rename it to reflect your new engine.
    ```shell
    cp path/to/runner/engines/vllm.py path/to/runner/engines/new_engine.py
    ```

3. **Customize the Engine Parameters**:
    - If your engine requires different initialization parameters, create or modify a parameter class similar to `VllmParams`. This class should inherit from `BaseModel` and define the necessary configuration for your model.
    ```python
    class NewEngineParams(BaseModel):
        # Define your parameters here
        model: str
        # ... other parameters ...
    ```

4. **Implement the New Engine Logic**:
    - Modify the `NewEngine` class to suit your requirements. Pay special attention to the `generate` method, as this is where the main logic for model interaction occurs.
    - Ensure to handle exceptions and edge cases effectively, as shown in the provided snippet.
    ```python
    class NewEngine(BaseEngine):
        def __init__(self, params: NewEngineParams):
            # Initialize your engine with the provided parameters

        @method()
        async def generate(self, payload: Payload, params):
            # Implement the generation logic for your new engine
            # This might include setting up asynchronous calls, handling streaming data, etc.
    ```

5. **Proceed to Container Creation**:
    - After successfully creating your new engine, the next step is to incorporate it into a new container. The container will provide the necessary runtime environment for your engine to execute. Follow the guidelines in the [add a new container](#adding-a-new-container) section to create a container that leverages your new engine.


>[!TIP]
> Developing a new engine requires an understanding of asynchronous programming in Python, especially for handling real-time data streaming and large-scale model interactions. If you're new to this, consider reviewing resources on async programming in Python and the specific libraries your project uses.

By following these detailed steps, contributors will have a clearer understanding of how to approach engine development, ensuring that new engines are robust, efficient, and well-integrated into the OpenRouter Runner system.

### Adding a new container

Adding a new container is necessary when your model requires a different environment, software, or hardware configuration. Follow these steps to create and integrate a new container:

1. **Copy an Existing Container**:
    - Use an existing container as your template.
    ```shell
    cp path/to/runner/containers/existing_container.py path/to/runner/containers/new_container.py
    ```

2. **Customize the Container**:
    - Modify the `new_container.py` file. Adjust the class name, Docker image, machine type, and any other settings to suit your new model's needs.
    ```python
    class NewContainer(BaseContainer):
        def __init__(self, ...):
            # Set up the new container's specific configurations.
    
        # Implement any additional methods required for the new container.
    ```

3. **Register the Container**:
    - In `./containers/__init__.py`, import your new container class and add your model's ID to the system.
    ```python
    from .new_container import NewContainer
    
    new_model_ids = ["your-new-model-id"]
    ```

4. **Include in Model Download List**:
    - Ensure the `all_models` list in the same `__init__.py` file includes your new list of model IDs.
    ```python
    all_models = [
        ...,
        *new_model_ids
    ]
    ```

5. **Testing Your Container**:
    - After setting up your new container, refer to the [Configuration and Testing](#configuration-and-testing) section to test it thoroughly.

>[!TIP]
> Creating a new container may require knowledge of Docker, cloud environments, and the specific needs of the model you're adding.

### Configuration and Testing

For detailed steps on setting up your environment and testing your models, please refer to the [Configuration and Testing section](../modal/runner/README.md#configuration-and-testing) in the Runner README. Here's a brief overview:

1. **Set Up Environment Variables**: Create a `.env.dev` file in the root of your project and include necessary details like `API_URL`, `RUNNER_API_KEY`, and `MODEL`.

2. **Install Dependencies**: Ensure all required tools and libraries are installed as per the instructions in the Runner README.

3. **Run Your Application for Testing**: Follow the steps to start your OpenRouter Runner and test the models using the provided scripts.

>[!NOTE]
> For comprehensive instructions, including how to load environment variables, choose test scripts, and interpret the results, please refer to the detailed guide in the [Runner README](../modal/runner/README.md#configuration-and-testing).

## Deploying

To deploy your OpenRouter Runner to Modal and monitor its performance, please see the [Deploying section](../modal/runner/README.md#deploying) in the Runner README. A summary of the steps includes:

1. **Deploy to Modal**: Use the `modal deploy runner` command to package your configurations and models into a live application.

2. **Monitor and Troubleshoot**: Keep an eye on your application's performance and logs through the Modal dashboard. Refer to the Runner README for detailed steps on how to access and interpret these logs.

>[!TIP]
> Always test your changes in a development environment before deploying to production. For a detailed guide on deployment best practices, refer to the [Runner README](../modal/runner/README.md#deploying).

## Contributions

Your contributions are invaluable to us. Please adhere to our [contributing guide](./CONTRIBUTING.md) and [code of conduct](./CODE_OF_CONDUCT.md) to maintain a healthy and welcoming community. For detailed instructions on adding models, engines, containers, and more, refer to the [Runner Setup Guide](../modal/runner/README.md).
