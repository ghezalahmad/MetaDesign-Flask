import pandas as pd

from app.utils.plot_generator import PlotGenerator


def test_tsne_one_hot_encodes_categorical_features():
    df = pd.DataFrame(
        {
            "water_binder_ratio": [0.34, 0.38, 0.42, 0.36, 0.46, 0.32],
            "slag_pct": [45, 35, 30, 50, 20, 55],
            "Fidelity_Level": ["standard", "standard", "standard", "high", "standard", "high"],
        }
    )

    prepared, encoded_columns, skipped_columns = PlotGenerator._prepare_tsne_feature_matrix(
        df,
        ["water_binder_ratio", "slag_pct", "Fidelity_Level"],
    )

    assert "Fidelity_Level" in encoded_columns
    assert skipped_columns == []
    assert "Fidelity_Level_high" in prepared.columns
    assert "Fidelity_Level_standard" in prepared.columns
    assert prepared.shape[0] == len(df)
    assert prepared.apply(lambda column: pd.api.types.is_numeric_dtype(column)).all()


def test_run_tsne_accepts_categorical_features():
    df = pd.DataFrame(
        {
            "water_binder_ratio": [0.34, 0.38, 0.42, 0.36, 0.46, 0.32],
            "slag_pct": [45, 35, 30, 50, 20, 55],
            "Fidelity_Level": ["standard", "standard", "standard", "high", "standard", "high"],
        }
    )

    result = PlotGenerator._run_tsne(
        df,
        ["water_binder_ratio", "slag_pct", "Fidelity_Level"],
        cache_key=None,
        perplexity=2,
        max_iter=250,
        random_state=7,
    )

    assert {"tsne-2d-one", "tsne-2d-two"}.issubset(result.columns)
    assert result["tsne-2d-one"].notna().all()
    assert result["tsne-2d-two"].notna().all()
