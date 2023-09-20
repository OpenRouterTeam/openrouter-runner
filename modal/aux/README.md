# Add new model

1. Copy one of the model file in `aux/models` and rename it to your model name.
2. Change the `model_id` to the HF model ID.
3. Adapt the Model class name accordingly.
4. Import the model into [](./models/__init__.py) and add it to the get_model function.

# Configuration

See [](./shared/common.py)

# Deployment

Make sure `pwd` is `/modal`, then:

```bash
modal deploy auxilary/main.py --env=dev

# OR
modal deploy auxilary/main.py --env=main
```
