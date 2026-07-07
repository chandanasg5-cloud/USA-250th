import pandas as pd

from src.model.city_cluster import RANKED_LABELS, cluster_cities


def _scores():
    # 8 synthetic cities in three obvious groups: two giants, three mids,
    # three smalls
    return pd.DataFrame({
        "city": list("ABCDEFGH"),
        "air_score": [100, 95, 50, 48, 52, 5, 3, 0],
        "events_score": [100, 90, 40, 45, 42, 2, 0, 5],
        "population_score": [100, 92, 55, 50, 45, 4, 6, 0],
        "income_score": [80, 85, 50, 55, 45, 10, 5, 0],
    })


def test_cluster_cities_deterministic_three_groups():
    out1, out2 = cluster_cities(_scores()), cluster_cities(_scores())
    assert out1.cluster_label.tolist() == out2.cluster_label.tolist()
    assert out1.cluster.nunique() == 3
    # the two giants share a cluster and get the top-ranked label
    a, b = out1.set_index("city").loc[["A", "B"], "cluster_label"]
    assert a == b == RANKED_LABELS[0]


def test_labels_are_plain_english_and_exhaustive():
    out = cluster_cities(_scores())
    assert set(out.cluster_label) == set(RANKED_LABELS)
