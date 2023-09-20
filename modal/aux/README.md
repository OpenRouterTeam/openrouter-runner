# Add new model

1. Copy one of the model file in `aux/models` and rename it to your model name.
2. Change the `model_id` to the HF model ID.
3. Adapt the Model class name accordingly.
4. Import the model into [](./models/__init__.py) and add it to the get_model function.

# Configuration

See [](./shared/common.py)

# Deployment

1.  Create a modal secret group
    HUGGINGFACE_TOKEN = <your huggingface token>
    with name "huggingface"
2.  Create a modal secret group
    AUX_API_KEY = <generate a random key>
    with name "ext-api-key"
3.  modal deploy aux/main.py
4.  Make sure `pwd` is `/modal`

5.  Run:

```bash
poetry shell

modal deploy auxilary/main.py --env=dev

# OR
modal deploy auxilary/main.py --env=main
```
