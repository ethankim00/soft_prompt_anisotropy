from sympy import re
from ansiotropy.embeddings.generate_embeddings import SoftPromptEmbeddingsExtractor

from ansiotropy.metrics import (
    get_average_mev,
    intra_sentence_cosine_similarity,
    inter_context_cosine_similarity,
    word_cosine_similarity,
)

# TODO add full experimental parameters

from typing import Dict
import pandas as pd


import pandas as pd
import wandb


def calculate_ansiotropy_metrics(
    model_id: str, model_type: str, datset_name: str, soft_token_num
) -> Dict:
    """
    Calculate the ansiotropy metrics for a given model and dataset

    Calulates the following metrics:
    1. Average Mean Maximum Explained Variance (AMEV)
    2. Intra-sentence cosine similarity
    3. Inter-context cosine similarity
    4. Word cosine similarity

    Calculate each metrics for both soft prompt and regular input tokens

    Args:
        model_id (str): id matching path to model file
        model_type (str): Autoregressive, Encoder Decoder, or Decoder ownly
        datset_name (str): SuperGLUE dataset name
        soft_token_num (int): Number of soft tokens

    Returns:
        Dict: Dictionary of metrics
    """
    extractor = SoftPromptEmbeddingsExtractor(model_path=model_id, dataset=datset_name)
    embeddings_dict = extractor.save_soft_prompt_embeddings(save_dict=False)
    soft_mev, regular_mev = get_average_mev(embeddings_dict, center=True)
    soft_intra_cos, regular_intra_cos = intra_sentence_cosine_similarity(
        embeddings_dict, center=True
    )
    soft_words_cos, regular_words_cos = word_cosine_similarity(
        embeddings_dict, center=True
    )
    results_dict = {
        "soft_mev": soft_mev,
        "regular_mev": regular_mev,
        "soft_intra_cos": soft_intra_cos,
        "regular_intra_cos": regular_intra_cos,
        "soft_words_cos": soft_words_cos,
        "regular_words_cos": regular_words_cos,
    }
    return results_dict


def compute_all_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all metrics for a given dataframe

    Iterate over the rows of the dataframe and call calculate_ansiotropy_metrics given the model_id and dataset_name
    Args:
        df (pd.DataFrame): Dataframe with wandb run data

    Returns:
        pd.DataFrame: Dataframe with metrics
    """
    metrics_list = []
    for index, row in df.iterrows():
        dictionary = calculate_ansiotropy_metrics(
            row["model_id"],
            row["model_type"],
            row["dataset_name"],
            row["soft_token_num"],
        )
        dictionary.update({"model_id": row["model_id"]})
        metrics_list.append(dictionary)
    metrics_df = pd.DataFrame(metrics_list)
    return metrics_df


def load_wandb_run(project_name) -> pd.DataFrame:
    """
    Load a Wandb run from api and save the config data to a datafrme
    Args:
        run_name (str): ID of wandb run

    Returns:
        pd.DataFrame: Dataframe with config data
    """
    api = wandb.Api()
    runs = api.runs(project_name)
    config_list = [{k: v for k, v in run.config.items()} for run in runs]
    df = pd.DataFrame(config_list)
    return df


if __name__ == "__main__":
    df = load_wandb_run("ethankim10/soft_prompt_anisotropy")
    metrics_df = compute_all_metrics(df)
    # merge metrics_df with df
    df = pd.merge(df, metrics_df, on="model_id")
    # save df to file
    df.to_csv("ansiotropy_metrics.csv")
    print(df.head())
