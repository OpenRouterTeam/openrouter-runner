import os
from typing import List

from shared.volumes import get_model_path, models_path

from .common import stub


def clean_models_volume(all_models: List[str], dry: bool = True):
    import shutil

    # go through each directory in the models volume
    for author in os.listdir(models_path):
        if author == "__cache__":
            continue
        print(f"Checking {author}")
        for model in os.listdir(models_path / author):
            model_id = f"{author}/{model}".lower()
            print(f"  {model_id} ...")
            # if the directory is not in all_models, delete it
            if model_id not in all_models:
                model_path = get_model_path(model_id)
                print(f"    Removing {model_path}")
                if not dry:
                    shutil.rmtree(model_path, ignore_errors=True)

    if not dry:
        stub.models_volume.commit()

    print("ALL DONE!")
