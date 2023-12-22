# OpenRouter Runner - Setup Guide

Welcome to the setup guide for the OpenRouter Runner. In this guide, you'll achieve the following:

- **Configure Your Environment**: Set up necessary tools and accounts.
- **Access Tokens**: Create and manage access tokens for secure interactions.
- **Prepare Containers and Engines**: Get your system ready to run a variety of models.
- **Deploy Models**: Learn how to add and deploy new models to the Runner.
- **Test and Monitor**: Understand how to test your deployments and monitor performance.

By the end of this guide, you'll have a comprehensive setup that allows you to manage and interact with advanced AI engines and access your models via API endpoints.

## Prerequisites

Before you begin, ensure you have the necessary accounts and tools:

1. **Modal Account**: Set up your environment on [Modal](https://modal.com/) as this will be your primary deployment platform.
2. **Hugging Face Account**: Obtain a token from [Hugging Face](https://huggingface.co/) for accessing models and libraries.

### Environment and Secrets Setup

Before you start working with the OpenRouter Runner, it's crucial to set up your working environment and secure access tokens. Follow these steps to ensure everything is properly configured:

1. **Set Current Directory**:
    First, make sure your current working directory is the `/modal` directory where your project is located. Adjust the path as needed for your setup.
    ```shell
    cd path/to/modal
    ```

2. **Create Development Environment**:
    If you haven't already set up a development environment in Modal, now is the time to do so. This provides a separate space where you can develop and test without affecting production. 

    ```shell
    modal environment create dev
    ```

    Now if you go to your dashboard on [Modal](https://modal.com/) you should be able to see the dev environment available in the environment drop down box. For easier development navigate to that environment so that you can track deployments and logging.

3. **Configure Modal Environment**:
    Switch to your development environment using the Modal CLI. This ensures all subsequent commands and deployments are done in the correct context.
    ```shell
    modal config set-environment dev # or main
    ```
    After running this command, you should see the 'dev' environment available in the environment dropdown box on your Modal dashboard.

    ![Modal Dev Environment](https://i.imgur.com/rSHguBw.png)
4. **Create Modal Secrets**:
    Securely store your API keys as secrets in Modal, which will be used by your application to access necessary resources and services.
    - **HuggingFace Token**:
      Create a modal secret group with your Hugging Face token. Replace `<your huggingface token>` with the actual token.
      ```shell
      modal secret create huggingface HUGGINGFACE_TOKEN=<your huggingface token>
      ```
    - **Runner API Key**:
      Create another secret group for the runner API key. Replace `<generate a random key>` with a strong, random key you've generated. Be sure to save this key somewhere as we'll need it for later!
      ```shell
      modal secret create ext-api-key RUNNER_API_KEY=<generate a random key>
      ```

 Now if you go to your dashboard on [Modal](https://modal.com/) and click on the secrets tab you should be able to see your keys deployed there.

 ![Modal Secret Keys Example](https://i.imgur.com/FSQr5lK.png)

## Getting Started with Deployment

With your environment now fully configured, you're ready to dive into deploying the OpenRouter Runner. This section guides you through deploying the Runner, adding new models or containers, and initiating tests to ensure everything is functioning as expected.

### Adding a New Model with Existing Containers

Adding new models to OpenRouter Runner is straightforward, especially when using models from Hugging Face that are compatible with existing containers. Here's how to do it:

1. **Find and Copy the Model ID**: Browse [Hugging Face](https://huggingface.co/models) for the model you wish to deploy. For example, let's use `"mistralai/Mistral-7B-Instruct-v0.2"`.

2. **Update Model List**: Open the `runner/containers/__init__.py` file. Locate the list of model IDs for the container you're using (e.g., `vllm_7b_model_ids` for a VLLM 7B model container). Add your new model ID to the list:
    ```python
    vllm_7b_model_ids = [
        "mistralai/Mistral-7B-Instruct-v0.1",
        "HuggingFaceH4/zephyr-7b-beta",
        "Intel/neural-chat-7b-v3-1",
        "Undi95/Toppy-M-7B",
        "mistralai/Mistral-7B-Instruct-v0.2" # New model added adhering to parameters
    ]
    ```
    This simple step registers the new model to be run with the existing container.

>[!IMPORTANT]
> When updating the `vllm_7b_model_ids`, ensure you uncomment the relevant model in the `all_models` list to allow it to be downloaded. For example:
> ```python
> all_models = [
>     # Uncomment the line below when adding new models to vllm_7b_model_ids
>     # *vllm_7b_model_ids,
>     *vllm_mid_model_ids,
>     *vllm_top_model_ids,
>     *vllm_a100_32k_model_ids,
>     *vllm_a100_128k_model_ids,
> ]
> ```

3. **Handle Access Permissions**: If you plan to deploy a model like `"meta-llama/Llama-2-13b-chat-hf"` which is included in the codebase, and you don't yet have access, visit [here](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) for instructions on how to request access. Temporarily, you can comment out this model in the list to proceed with deployment.

4. **Deploy the Model**: Run the following command to download the models and deploy the containerized app to Modal:
    ```shell
    modal run runner::download
    ```
    This command pulls the specified models and packages them into your configured containers for deployment. Now if you go to your dashboard on [Modal](https://modal.com/) you'll be able to see that your app is running and downloaded and the models. As a heads up it might take a few minutes to download all the models. 

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

5. **Associate Models**: In the same `__init__.py` file, make sure the `all_models` list includes your new list of model IDs, so they are recognized and downloaded by the system.
    ```python
    all_models = [
        *new_model_ids,
        # Include other model lists as necessary.
    ]
    ```

6. **Deploy the Container**: Use the Modal CLI to deploy your new container. This will download the required models and deploy your containerized app to Modal.
    ```shell
    modal run runner::download
    ```
    After running this command, check your Modal dashboard to confirm that your app is running and the models are downloaded. Be patient; downloading all the models can take some time.

7. **Start Testing**: With your new container deployed, proceed to the [Configuration and Testing](#configuration-and-testing) section to begin testing your model!

>[!NOTE]
> Creating a new container can be complex and requires a good understanding of the model's needs and the system's capabilities. If you encounter difficulties, consult the detailed documentation, or seek support from the community or help forums.

With these steps, you're well on your way to enhancing the OpenRouter Runner with a wider array of models and capabilities. For any assistance or further customization, refer to our [contributing guide](./.github/CONTRIBUTING.md).


<a name="configuration-and-testing"></a>
## Configuration and Testing

Before diving into testing your models and endpoints, it's essential to properly configure your environment and install all necessary dependencies. This section guides you through setting up your environment, running test scripts, and ensuring everything is functioning correctly.

### Setting Up Your Environment

1. **Create a `.env.dev` File**: In the root of your project, create a `.env.dev` file to store your environment variables. This file should include:
    ```plaintext
    API_URL=<MODAL_API_ENDPOINT_THAT_WAS_DEPLOYED>
    RUNNER_API_KEY=<CUSTOM_KEY_YOU_CREATED_EARLIER>
    MODEL=<MODEL_YOU_ADDED_OR_WANT_TO_TEST>
    ```
    - `API_URL`: Your endpoint URL, obtained after deploying the runner and downloading the models.
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
    - Ensure the `huggingface_hub` module is installed in your Python environment. This is crucial for interacting with models from Hugging Face.
        ```shell
        pip install huggingface_hub
        ```

### Running Your App for Testing

1. **Ensure the Runner is Active**: Make sure your OpenRouter Runner is running. From the `openrouter-runner/modal` directory, you can start it with:
    ```shell
    modal serve runner
    ```
    This command will keep your app running and ready for testing.

2. **Open Another Terminal for Testing**: While keeping the runner active, open a new terminal window. Navigate to the `/openrouter-runner` path to be in the correct directory for running scripts.

### Testing a Model

Now that your environment is set up and your app is running, you're ready to start testing models.

1. **Navigate to Project Root**: Ensure you're in the root directory of your project.
    ```shell
    cd path/to/openrouter-runner
    ```

2. **Load Environment Variables**: Source your `.env.dev` file to load the environment variables.
    ```shell
    source .env.dev
    ```

3. **Choose a Test Script**: In the `scripts` directory, you'll find various scripts for testing different aspects of your models. For a simple test, you might start with `test-simple.ts`.

4. **Run the Test Script**: Execute the script with your model identifier using the command below. Replace `YourModel/Identifier` with the specific model you want to test.
    ```shell
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
    ```shell
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

Interested in contributing? Please read our [contributing guide](./.github/CONTRIBUTING.md) and follow our [code of conduct](./.github/CODE_OF_CONDUCT.md).
