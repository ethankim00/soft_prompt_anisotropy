### Calculate relevant metrics for ansiotropy of contextual embeddings
# 1. Inter Similarity
# 2. Intra Similarity
# 3. MEV
# 4. Local intrinsic dimension
# 5. I(W) Metric w Mu and Viswanath (2018)


from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, normalize

import numpy as np


def inter_context_cosine_similarity(
    X: np.ndarray, center: bool = False, sample_size: int = 0
) -> float:
    if center:
        X = StandardScaler(with_std=False).fit(X).transform(X)
    if sample_size >= 0:
        X = X.sample(sample_size)
    cos = cosine_similarity(X, X)  # pairwise cosine similarities
    avg_cos = (
        (np.sum(np.sum(cos)) - cos.shape[0])
        / 2
        / (cos.shape[0] * (cos.shape[0] - 1) / 2)
    )
    return avg_cos


def maximum_explainable_variance(X: np.ndarray, num_components: int = 1) -> float:
    pca = PCA(n_components=num_components)
    pca.fit(X)
    explained_variance = sum(pca.explained_variance_ratio_)
    return explained_variance
