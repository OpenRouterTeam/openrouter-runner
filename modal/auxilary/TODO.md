- Refactor model to spawn dynamic image

```python
image = mk_gpu_image("PygmalionAI/mythalion-13b")

# ERROR:
#   Modal can only import functions defined in global scope unless they are `serialized=True`
```

- Refactor model template to reuse class

```python
from auxilary.models.mythalion_13b import (
    Model as Mythalion13BModel,
    model_id as mythalion_13b_model_id,
)

from auxilary.models.mythomax_13b import (
    Model as Mythomax13BModel,
    model_id as mythomax_13b_model_id,
)

# Warning: Tag 'Model.tokenize_prompt' collision! Overriding existing function [auxilary.models.mythalion_13b].Model.tokenize_prompt with new function [auxilary.models.mythomax_13b].Model.tokenize_prompt
```
