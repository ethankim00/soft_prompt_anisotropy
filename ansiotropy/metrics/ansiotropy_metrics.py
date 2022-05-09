### Calculate relevant metrics for ansiotropy of contextual embeddings
# 1. Inter Similarity

# 2. Intra Similarity
# 3. MEV
# 4. Local intrinsic dimension
# 5. I(W) Metric w Mu and Viswanath (2018)
from itertools import chain
import pickle
from typing import Dict
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, normalize

import numpy as np


def center_token_embeddings(embeddings: Dict) -> Dict:
    """Center the token embeddings.

    Args:
        embeddings (Dict): Dictionary of Sentences and Embeddings

    Returns:
        Dict: Dictionary of Sentences and Embeddings with centered token embeddings
    """
    scaler = StandardScaler(with_std=False)
    all_embeddings = np.concatenate(
        [list(sentence.values())[0] for sentence in embeddings["sentences"]]
    )
    scaler.fit(all_embeddings)
    embeddings["sentences"] = [
        {k: scaler.transform(v) for k, v in sentence.items()}
        for sentence in embeddings["sentences"]
    ]
    return embeddings


def word_cosine_similarity(embeddings_dict: Dict, center: bool = False) -> float:
    """Calculate the average cosine similarity between all word embeddings

    Args:
        embeddings (Dict): Dictionary of Sentences and Embeddings
        center (bool, optional): Whether to center the embeddings. Degfaults to False.

    Returns:
        float: Average cosine similarity between all word embeddings
    """
    if center:
        embeddings_dict = center_token_embeddings(embeddings_dict)
    similarities = []
    similarities_dict = {}
    for token, embedding in embeddings_dict["tokens"].items():
        embedding = np.vstack(embedding)
        cos = inter_context_cosine_similarity(embedding)
        similarities.append(cos)
        similarities_dict[token] = cos
    avg_cos = np.mean(similarities)
    avg_cos_soft = np.mean(similarities[:embeddings_dict["soft_prompt_tokens"]])
    avg_cos_regular = np.mean(similarities[embeddings_dict["soft_prompt_tokens"]])
    print("Avg cosine similarity", avg_cos)
    print("Soft cosine similarity", avg_cos_soft)
    print("Regular cosine similarity", avg_cos_regular)
    return avg_cos, similarities_dict


def inter_context_cosine_similarity(
    X: np.ndarray, center: bool = False, sample_size: int = 0
) -> float:
    if center:
        X = StandardScaler(with_std=False).fit(X).transform(X)
    # if sample_size >= 0:
    #    X = X.sample(sample_size)
    cos = cosine_similarity(X, X)  # pairwise cosine similarities
    avg_cos = (
        (np.sum(np.sum(cos)) - cos.shape[0])
        / 2
        / (cos.shape[0] * (cos.shape[0] - 1) / 2)
    )
    return avg_cos


def maximum_explainable_variance(
    X: np.ndarray, center: bool = False, num_components: int = 1
) -> float:
    """Calculate the maximum explainable variance for a given embedding matrix."""
    if center:
        X = StandardScaler(with_std=False).fit(X).transform(X)
    pca = PCA(n_components=num_components)
    pca.fit(X)
    explained_variance = sum(pca.explained_variance_ratio_)
    return explained_variance


def get_average_mev(embeddings_dict: Dict, center: bool = False):
    if center:
        embedding_dict = center_token_embeddings(embeddings_dict)
    mev_dict = {}
    mev_list = []
    for token, embedding in embedding_dict["tokens"].items():
        embedding = np.vstack(embedding)
        mev = maximum_explainable_variance(embedding)
        mev_dict[token] = mev
        mev_list.append(mev)
    return np.mean(mev_list), mev_dict


def intra_sentence_cosine_similarity(embeddings: Dict, center: bool = False) -> float:
    """Calculate the average cosine similarity between a word and its sentence representation.
    The sentence representation is the average of all word embeddings in the sentence.

    Args:
        embeddings (Dict): Dictionary of Sentences and Embeddings
        center (bool, optional): Whether to center the embeddings. Defaults to False.

    Returns:
        float: Average cosine similarity between token embeddings in the same sentence
    """
    if center:
        scaler = StandardScaler(with_std=False)
        all_embeddings = np.concatenate(
            [list(sentence.values())[0] for sentence in embeddings["sentences"]]
        )
        scaler.fit(all_embeddings)
        embeddings["sentences"] = [
            {k: scaler.transform(v) for k, v in sentence.items()}
            for sentence in embeddings["sentences"]
        ]
    similarities = []
    soft_similarities = []
    regular_similarities = []
    for sentence in embeddings["sentences"]:
        embedding = list(sentence.values())[0]
        sentence_emdedding = np.mean(embedding, axis=0).reshape(1, -1)
        sentence_cos = cosine_similarity(sentence_emdedding, embedding)
        avg_cos = np.mean(sentence_cos)
        if embeddings["soft_prompt_tokens"] > 0:
            avg_cos_soft = np.mean(sentence_cos[:, :embeddings["soft_prompt_tokens"]])
        else:
            avg_cos_soft = 0
        avg_cos_regular = np.mean(sentence_cos[:, embeddings["soft_prompt_tokens"]:])
        similarities.append(avg_cos)
        soft_similarities.append(avg_cos_soft)
        regular_similarities.append(avg_cos_regular)
    print("soft", np.mean(soft_similarities))
    print("regular", np.mean(regular_similarities))
    return np.mean(similarities)


if __name__ == "__main__":
    embedding_dict = pickle.load(
        open(Path("./data/embeddings").joinpath("embeddings_20220424-164618.pkl"), "rb")
    )
    # X = np.vstack(embedding_dict["tokens"]["soft_token_0"])
    # print(inter_context_cosine_similarity(X, center=True))
    # self_sim, sim_dict = word_cosine_similarity(embedding_dict, center=True)
    # print(self_sim)
    # print(sim_dict)
    # print(intra_sentence_cosine_similarity(embedding_dict, center=False))
    # print(get_average_mev(embedding_dict, center=True))
