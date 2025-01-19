# OpenRouter Runner

OpenRouter Runner is a monolith inference engine, built with [Modal](https://modal.com/). It serves as a robust solution for the deployment of tons of open source models that are hosted in a fallback capacity on [openrouter.ai](https://openrouter.ai).

> ✨ If you can make the Runner run faster and cheaper, we'll route to your services!

#### Table of Contents
- [Adding Models To OpenRouter (Video)](#adding-models-to-openrouter)
- [Prerequisites](#prerequisites)
- [Quickstart](#quickstart)
- [Adding New Models](#adding-new-models)
  - [Adding a New Model With Existing Containers](#adding-a-new-model-with-existing-containers)
  - [Adding a New Model With New Container](#adding-a-new-model-requiring-a-new-container)
- [Configuration and Testing](#configuration-and-testing)
- [Deploying](#deploying)
- [Contributions](#contributions)


# Adding Models To OpenRouter
[![Watch the video](https://img.youtube.com/vi/Ob9xx44Gb_o/maxresdefault.jpg)](https://youtu.be/Ob9xx44Gb_o)


# Prerequisites

Before you begin, ensure you have the necessary accounts and tools:

1. **Modal Account**: Set up your environment on [Modal](https://modal.com/) as this will be your primary deployment platform.
2. **Hugging Face Account**: Obtain a token from [Hugging Face](https://huggingface.co/) for accessing models and libraries.
3. **Poetry Installed**: Make sure you have [poetry](https://python-poetry.org/docs/) installed on your machine.

# Quickstart

For those familiar with the OpenRouter Runner and wanting to deploy it quickly. This means you have already set up the [prerequisites](#prerequisites) and can start deploying.

1. **Navigate to modal directory.**

    ```shell
    cd path/to/modal
    ```

2. **Setup Poetry**

    ```sh
    poetry install
    poetry shell
    modal token new
    ```

    > ℹ️ For intellisense, it's recommended to run vscode via the poetry shell:

    ```sh
    poetry shell
    code .
    ```

3. **Create dev environment**
    
    ```shell Python
    modal environment create dev
    ```

    > ℹ️ If you have a dev environment created already no need to create another one. Just configure to it in the next step.

4. **Configure dev environment**

    ```shell Python
    modal config set-environment dev
    ``` 
    > ⚠️ We are using our Dev environment right now. Switch to **main** when deploying to production.


5. **Configure secret keys**

    - **HuggingFace Token**:
      Create a Modal secret group with your Hugging Face token. Replace `<your huggingface token>` with the actual token.
      ```shell Python
      modal secret create huggingface HUGGINGFACE_TOKEN=<your huggingface token>
      ```
    - **Runner API Key**:
      Create a Modal secret group for the runner API key. Replace `<generate a random key>` with a strong, random key you've generated. Be sure to save this key somewhere as we'll need it for later!
      ```shell Python
      modal secret create ext-api-key RUNNER_API_KEY=<generate a random key>
      ```
    
    - **Sentry Configuration** 
      Create a Modal secret group for the Sentry error tracking storage. Replace `<optional SENTRY_DSN>` with your DSN from sentry.io or leave it blank to disable Sentry (e.g. `SENTRY_DSN=`). You can also add an environment by adding `SENTRY_ENVIRONMENT=<environment name>` to the command.
      ```shell Python
      modal secret create sentry SENTRY_DSN=<optional SENTRY_DSN>
      ```
    
    - **Datadog Configuration** 
      Create a Modal secret group for Datadog log persistence. Replace `<optional DD_API_KEY>` with your Datadog API Key or leave it blank to disable Datadog (e.g. `DD_API_KEY=`). You can also add an environment by adding `DD_ENV=<environment name>` to the command and a site by adding `DD_SITE=<site name>` to the command.
      ```shell Python
      modal secret create datadog DD_API_KEY=<optional DD_API_KEY> DD_SITE=<site name>
      ```
    
  6. **Download Models**
     
      ```shell Python
      modal run runner::download
      ```

  7. **Deploy Runner**

      ```shell Python
      modal deploy runner
      ```

## Adding New Models

With your environment now fully configured, you're ready to dive into deploying the OpenRouter Runner. This section guides you through deploying the Runner, adding new models or containers, and initiating tests to ensure everything is functioning as expected.

### Adding a New Model with Existing Containers

Adding new models to OpenRouter Runner is straightforward, especially when using models from Hugging Face that are compatible with existing containers. Here's how to do it:

1. **Find and Copy the Model ID**: Browse [Hugging Face](https://huggingface.co/models) for the model you wish to deploy. For example, let's use `"mistralai/Mistral-7B-Instruct-v0.2"`.

2. **Update Model List**: Open the `runner/containers/__init__.py` file. Add your new model ID to the `DEFAULT_CONTAINER_TYPES` dictionary, using the container definition you want to use:
    ```python
    DEFAULT_CONTAINER_TYPES = {
        "Intel/neural-chat-7b-v3-1": ContainerType.VllmContainer_7B,
        "mistralai/Mistral-7B-Instruct-v0.2": ContainerType.VllmContainer_7B,
        ...
    }
    ```


3. **Handle Access Permissions**: If you plan to deploy a model like `"meta-llama/Llama-2-13b-chat-hf"`, and you don't yet have access, visit [here](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) for instructions on how to request access. Temporarily, you can comment out this model in the list to proceed with deployment.

4. **Download and Prepare Models**: Use the CLI to execute the `runner::download` function within your application. This command is designed to download and prepare the required models for your containerized app.
    ```shell Python
    modal run runner::download
    ```
    This step does not deploy your app but ensures all necessary models are downloaded and ready for when you do deploy. After running this command, you can check the specified storage location or logs to confirm that the models have been successfully downloaded. Note that depending on the size and number of models, this process can take some time.

5. **Start testing the Models**: Now you can go to the [Configuration and Testing](#configuration-and-testing) section to start testing your models!

### Adding a New Model Requiring a New Container

Sometimes the model you want to deploy requires an environment or configurations that aren't supported by the existing containers. This might be due to special software requirements, different machine types, or other model-specific needs. In these cases, you'll need to create a new container.

1. **Understand the Requirements**: Before creating a new container, make sure you understand the specific requirements of your model. This might include special libraries, hardware needs, or environment settings.

2. **Copy a Container File**: Start by copying an existing container file from `runner/containers`. This gives you a template that's already integrated with the system.
    ```shell 
    cp runner/containers/existing_container.py runner/containers/new_container.py
    ```

3. **Customize the Container**: Modify the new container file. Change the class name to something unique, and adjust the image, machine type, engine, and any other settings to meet your model's needs. Remember to install any additional libraries or tools required by your model.

4. **Register the Container**: Open `./containers/__init__.py`. Add an import statement for your new container class at the top of the file, then create a new list of model IDs or update an existing one to include your model.
    ```python
    from .new_container import NewContainerClass

    new_model_ids = [
        "your-model-id",
        # Add more model IDs as needed.
    ]
    ```

5. **Associate Models**: Add a `ContainerType` for your model in `modal/shared/protocol.py` and define how to build it in `get_container(model_path: Path, container_type: ContainerType)` in `modal/runner/containers/__init__.py`.

6. **Download and Prepare Models**: Use the CLI to execute the `runner::download` function within your application. This command is designed to download and prepare the required models for your containerized app.
    ```shell Python
    modal run runner::download
    ```
    This step does not deploy your app but ensures all necessary models are downloaded and ready for when you do deploy. After running this command, you can check the specified storage location or logs to confirm that the models have been successfully downloaded. Note that depending on the size and number of models, this process can take some time.


7. **Start Testing**: With your new container deployed, proceed to the [Configuration and Testing](#configuration-and-testing) section to begin testing your model!

>[!NOTE]
> Creating a new container can be complex and requires a good understanding of the model's needs and the system's capabilities. If you encounter difficulties, consult the detailed documentation, or seek support from the community or help forums.

## Configuration and Testing

Before diving into testing your models and endpoints, it's essential to properly configure your environment and install all necessary dependencies. This section guides you through setting up your environment, running test scripts, and ensuring everything is functioning correctly.

### Setting Up Your Environment

1. **Create a `.env.dev` File**: In the root of your project, create a `.env.dev` file to store your environment variables. This file should include:
    ```plaintext
    API_URL=<MODAL_API_ENDPOINT_THAT_WAS_DEPLOYED>
    RUNNER_API_KEY=<CUSTOM_KEY_YOU_CREATED_EARLIER>
    MODEL=<MODEL_YOU_ADDED_OR_WANT_TO_TEST>
    ```
    - `API_URL`: Your endpoint URL, obtained downloading the models. You can find this on your Modal dashboard as well.
    - `RUNNER_API_KEY`: The custom key you created earlier.
    - `MODEL`: The identifier of the model you wish to test.

2. **Install Dependencies**:
If you haven't already install the following dependencies.

    - If you're working with TypeScript scripts, you'll likely need to install Node.js packages. Use the appropriate package manager for your project:
        ```shell 
        npm install
        # or
        pnpm install
        ```

### Running Your App for Testing

1. **Ensure the Runner is Active**: Make sure your OpenRouter Runner is running. From the `openrouter-runner/modal` directory, you can start it with:
    ```shell Python
    modal serve runner
    ```
    This command will keep your app running and ready for testing.

2. **Open Another Terminal for Testing**: While keeping the runner active, open a new terminal window. Navigate to the `/openrouter-runner` path to be in the correct directory for running scripts.

### Testing a Model

Now that your environment is set up and your app is running, you're ready to start testing models.

1. **Navigate to Project Root**: Ensure you're in the root directory of your project.
    ```shell Python
    cd path/to/openrouter-runner
    ```

2. **Load Environment Variables**: Source your `.env.dev` file to load the environment variables.
    ```shell Python
    source .env.dev
    ```

3. **Choose a Test Script**: In the `scripts` directory, you'll find various scripts for testing different aspects of your models. For a simple test, you might start with `test-simple.ts`.

4. **Run the Test Script**: Execute the script with your model identifier using the command below. Replace `YourModel/Identifier` with the specific model you want to test.
    ```shell Python
    pnpm x scripts/test-simple.ts YourModel/Identifier
    ```
>[!NOTE]
> If you wish to make the results more legible, especially for initial tests, consider setting `stream: false` in your script to turn off streaming.

5. **Viewing Results**: After running the script, you'll see a JSON-formatted output in your terminal. It will provide the generated text along with information on the number of tokens used in the prompt and completion. If you've set `stream: false`, the text will be displayed in its entirety, making it easier to review the model's output.

    **Example Response**:
    ```json
    {
      "text": "Project A119 was a top-secret program run by the United States government... U.S. nuclear and military policies.",
      "prompt_tokens": 23,
      "completion_tokens": 770,
      "done": true
    }
    ```
    *Note: The response has been truncated for brevity.*

6. **Troubleshooting**: If you encounter errors related to Hugging Face models, ensure you've installed `huggingface_hub` and have the correct access permissions for the models you're trying to use.

By following these steps, you should be able to set up your environment, deploy your app, and start testing various models and endpoints. Remember to consult the detailed documentation and seek support from the community if you face any issues.

## Deploying

Deploying your model is the final step in making your AI capabilities accessible for live use. Here's how to deploy and what to expect:

1. **Deploy to Modal**:
    When you feel confident with your setup and testing, deploy your runner to Modal with the following command:
    ```shell Python
    modal deploy runner
    ```
    This command deploys your runner to Modal, packaging your configurations and models into a live, accessible application.

2. **View Your Deployment**:
    After deployment, visit your dashboard on [Modal](https://modal.com/). You should see your newly deployed model listed there. This dashboard provides useful information and controls for managing your deployment.

3. **Interact with Your Live Model**:
    With your model deployed, you can now call the endpoints live. Use the API URL provided in your `.env.dev` file (or found on your Modal dashboard) to send requests and receive AI-generated responses. This is where you see the real power of your OpenRouter Runner in action.

4. **Monitor and Troubleshoot**:
    Keep an eye on your application's performance and logs through the Modal dashboard. If you encounter any issues or unexpected behavior, consult the logs for insights and adjust your configuration as necessary.

By following these steps, your OpenRouter Runner will be live and ready to serve!

## Contributions

We'd love to see you add more models to the Runner! If you're interested in contributing, please follow the section on [Adding a New Model](#adding-new-models) to start adding more Open Source models to OpenRouter! In addition, please adhere to our [code of conduct](./CODE_OF_CONDUCT.md) to maintain a healthy and welcoming community.
