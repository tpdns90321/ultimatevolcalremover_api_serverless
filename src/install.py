from uvr.utils.get_models import download_all_models
import json

models_json = json.load(
    open("../ultimatevocalremover_api/src/models_dir/models.json", "r")
)
download_all_models(models_json)
