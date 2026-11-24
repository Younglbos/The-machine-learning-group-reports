"""K-means with a fixed, explicit restart and stopping protocol."""

import os

# The course runner uses one CPU thread; skip Windows WMIC core discovery.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
from sklearn.cluster import KMeans


def fit_kmeans(features, clusters, seed, options, centers=None):
    model = KMeans(n_clusters=clusters, random_state=seed,
                   n_init=options["n_init"] if centers is None else 1,
                   init="k-means++" if centers is None else centers,
                   max_iter=options["max_iter"], tol=options["tol"], algorithm="lloyd")
    predicted = model.fit_predict(features)
    return model, predicted
