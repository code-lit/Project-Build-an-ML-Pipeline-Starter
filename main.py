import json

import mlflow
import tempfile
import os
import wandb
import hydra
from omegaconf import DictConfig

_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "train_val_test_split",
    "train_random_forest",
    # NOTE: We do not include this in the steps so it is not run by mistake.
    # You first need to promote a model export to "prod" before you can run this,
    # then you need to run this step explicitly
   "test_regression_model"
]


# This automatically reads in the configuration
@hydra.main(version_base=None, config_name='config', config_path='.')
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:

        if "download" in active_steps:
            # Download file and load in W&B
            _ = mlflow.run(
                f"{config['main']['components_repository']}/get_data",
                "main",
                env_manager="conda",
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded"
                },
            )

        if "basic_cleaning" in active_steps:
            _ = mlflow.run(
                f"{config['main']['components_repository']}/basic_cleaning",
                "main",
                env_manager="conda",
                parameters={
                    "input_artifact": config["data"]["raw_artifact"],
                    "output_artifact": "clean_data.csv",
                    "output_type": "clean_data",
                    "output_description": "Cleaned dataset after basic preprocessing"
                },
            )

        pass

        if "data_check" in active_steps:
            _ = mlflow.run(
                f"{config['main']['components_repository']}/data_check",
                "main",
                env_manager="conda",
                parameters={
                    # These must match src/data_check/MLproject parameter names:
                    "csv": "clean_sample.csv:latest",
                    "ref": "clean_sample.csv:reference",
                    "kl_threshold": config["data_check"]["kl_threshold"],
                    "min_price": config["etl"]["min_price"],
                    "max_price": config["etl"]["max_price"],
                },
            )

        pass

        if "train_val_test_split" in active_steps:
            _ = mlflow.run(
                f"{config['main']['components_repository']}/train_val_test_split",
                "main",
                env_manager="conda",
                parameters={
                    "input": "clean_sample.csv:latest",
                    "test_size": str(config["modeling"]["test_size"]),
                    "random_seed": str(config["modeling"]["random_seed"]),
                    "stratify_by": str(config["modeling"]["stratify_by"]),
                },
            )

        pass

        if "train_random_forest" in active_steps:
            # Write rf_config.json from config.yaml
            rf_config = os.path.abspath("rf_config.json")
            with open(rf_config, "w+") as fp:
                json.dump(dict(config["modeling"]["random_forest"].items()), fp)

            _ = mlflow.run(
                "src/train_random_forest",
                "main",
                env_manager="conda",
                parameters={
                    # Input artifact produced by train_val_test_split
                    "trainval_artifact": "trainval_data.csv:latest",

                    # Model training controls (from config.yaml)
                    "val_size": float(config["modeling"]["val_size"]),
                    "random_seed": int(config["modeling"]["random_seed"]),
                    "stratify_by": str(config["modeling"]["stratify_by"]),
                    "max_tfidf_features": int(config["modeling"]["max_tfidf_features"]),

                    # Random Forest hyperparameters JSON written above
                    "rf_config": rf_config,

                    # Output artifact name (as required)
                    "output_artifact": "random_forest_export",
                },
            )

        pass

        if "test_regression_model" in active_steps:
            _ = mlflow.run(
                "components/test_regression_model",
                "main",
                env_manager="conda",
                parameters={
                    # model in W&B promoted to prod
                    "mlflow_model": "random_forest_export:prod",

                    # test split artifact from train_val_test_split
                    "test_dataset": "test_data.csv:latest"


                },
            )

        pass

if __name__ == "__main__":
    go()
