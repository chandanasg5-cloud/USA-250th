"""KMeans segmentation of the focus cities (k=3, illustrative).

8 observations is a demo of the technique, not inference — the dashboard
caption says so. Labels are plain English, assigned deterministically by
ranking cluster centroids on the default-weight composite.
"""
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.model.city_index import DEFAULT_WEIGHTS, INDEX_PATH, composite

FEATURES = ["air_score", "events_score", "population_score", "income_score"]
RANKED_LABELS = ["National anniversary magnets", "Strong regional draws",
                 "Focused local hosts"]


def cluster_cities(scores: pd.DataFrame, k: int = 3,
                   seed: int = 42) -> pd.DataFrame:
    X = StandardScaler().fit_transform(scores[FEATURES])
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(X)
    out = scores.copy()
    out["cluster"] = km.labels_
    rank = (out.assign(_c=composite(out, DEFAULT_WEIGHTS))
            .groupby("cluster")._c.mean()
            .sort_values(ascending=False))
    out["cluster_label"] = out.cluster.map(
        {cl: RANKED_LABELS[i] for i, cl in enumerate(rank.index)})
    return out


def main() -> None:
    scores = pd.read_csv(INDEX_PATH)
    out = cluster_cities(scores)
    out.to_csv(INDEX_PATH, index=False)
    print(f"clusters: {out.groupby('cluster_label').city.count().to_dict()}")


if __name__ == "__main__":
    main()
