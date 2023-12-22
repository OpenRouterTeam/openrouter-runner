# OpenRouter Runner

OpenRouter Runner is a monolith inference engine, built with [Modal](https://modal.com/) a platform for deploying scalable apps. It serves as a robust solution for lots of the open source models hosted in a fallback capacity on [openrouter.ai](https://openrouter.ai).

## Runner Structure

The OpenRouter Runner consists of three main components that can be scaled out and customized.
- **Containers** 
- **Engines**
- **Endpoints** 

The types of AI models that are available to be used in the Runner are dictated by the defined Containers and Engines. If you are interested in adding to the runner please read our [contributing guide](./.github/CONTRIBUTING.md) and follow our [code of conduct](./.github/CODE_OF_CONDUCT.md).

### Containers

[Containers](https://cloud.google.com/learn/what-are-containers) are at the core of the OpenRouter Runner, prepared for deployment on [Modal](https://modal.com/), a cloud platform for running scalable containerized jobs. They enable efficient use of various inference engines.

- **Flexible Environments**: Designed to support various container bases with the necessary libraries for diverse AI models.

- **Configurable Resources**: Allows for tailored GPU and memory settings to match engine demands.

- **High Throughput**: Engineered for concurrent processing, ensuring multiple requests are handled swiftly.

- **Distributed Efficiency**: Integrates with Modal's distributed computing to scale across GPUs seamlessly.

- **Cloud-Native Deployment**: Simplified deployment process, making it accessible through a command or web interface.

For deployment instructions and engine integration on Modal, visit our [deployment guide](link-to-deployment-guide).


### Engines

[Engines](https://www.autoblocks.ai/glossary/inference-engine) in the OpenRouter Runner are responsible for executing model inference, which is the process of deriving predictions or decisions from a trained machine learning model. OpenRouter Runner supports a variety of engines, each optimized for different types of models or tasks.

- **vLLM (Very Large Language Models)**: These engines are designed to handle the most advanced and sizeable language models, providing the computational power necessary to process extensive natural language data at scale.

- **HF Transformers**: Built on the widely-used Hugging Face Transformers library, these engines provide a seamless experience for deploying transformer-based models, which are essential for a wide range of NLP tasks from text classification to question answering.

- **Custom Engines**: OpenRouter Runner is engineered to be extensible, allowing developers to integrate their own custom-built engines. Whether you have a specialized use case or require unique processing capabilities, our system is designed to accommodate your needs.

For instructions on how to deploy an additional engine to the Runner, check out our [contributing guide](./.github/CONTRIBUTING.md).

### Endpoints

In OpenRouter Runner, **Endpoints** are the gateways through which users interact with the various AI models and engines. They are the accessible URLs or URIs that accept input (like text) and return the AI-generated results. Here's how endpoints can diversify the capabilities of OpenRouter Runner:

- **Completion Endpoint**: Currently, we have a completion endpoint that handles text-based requests and returns generated text completions. It's ideal for applications like chatbots, text completion, and creative writing assistance.

- **Custom Endpoints**: OpenRouter Runner is designed with extensibility in mind. Developers can create custom endpoints for other models that include items like Image Generation, Text-to-speech, and more.

As we continue to develop and expand the OpenRouter Runner, new endpoints will be added to meet the evolving needs of our users. Stay tuned for updates, and if you have suggestions please let us know in the [OpenRouter Discord](https://discord.com/channels/1091220969173028894/1107397803266818229).


## Getting Started

If you're interested in building on top of the OpenRouter Runner follow the instructions below to get started.

1. Fork the OpenRouter Runner Repository
2. In your code editor of choice go to the modal folder
    `cd modal`
3. Follow the instructions in the [Runner ReadMe](./modal/runner/README.md) to get started developing.

## Contributions

Interested in contributing? Please read our [contributing guide](./.github/CONTRIBUTING.md) and follow our [code of conduct](./.github/CODE_OF_CONDUCT.md).

## License

[MIT](./LICENSE)
