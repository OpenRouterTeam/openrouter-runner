from aux.models.mythalion_13b import (
    Mythalion13BModel,
    model_id as mythalion_13b_model_id,
)

from aux.models.mythomax_13b import (
    Mythomax13BModel,
    model_id as mythomax_13b_model_id,
)

from aux.models.remm_slerp_13b import (
    RemmSlerp13BModel,
    model_id as remm_slerp_13b_model_id,
)
from aux.models.llama2_chat_13b import (
    Llama2Chat13BModel,
    model_id as llama2_chat_13b_model_id,
)

from aux.models.nous_hermes_13b import (
    NousHermes13BModel,
    model_id as nous_hermes_13b_model_id,
)


def get_model(model: str):
    normalized_model_id = model.lower()
    if normalized_model_id == mythalion_13b_model_id.lower():
        return Mythalion13BModel()
    elif normalized_model_id == mythomax_13b_model_id.lower():
        return Mythomax13BModel()
    elif normalized_model_id == remm_slerp_13b_model_id.lower():
        return RemmSlerp13BModel()
    elif normalized_model_id == llama2_chat_13b_model_id.lower():
        return Llama2Chat13BModel()
    elif normalized_model_id == nous_hermes_13b_model_id.lower():
        return NousHermes13BModel()
    else:
        raise ValueError(f"Invalid model: {model}")
