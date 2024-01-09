import os
from typing import List

from shared.volumes import get_model_path, models_path, models_volume


def clean_models_volume(all_models: List[str], dry: bool = True):
    import shutil

    remove_all = len(all_models) == 0
    # go through each directory in the models volume
    for author in os.listdir(models_path):
        if author == "__cache__":
            if remove_all:
                shutil.rmtree(get_model_path(author), ignore_errors=True)
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
        if remove_all:
            shutil.rmtree(get_model_path(author), ignore_errors=True)
    if not dry:
        models_volume.commit()

    print("ALL DONE!")
