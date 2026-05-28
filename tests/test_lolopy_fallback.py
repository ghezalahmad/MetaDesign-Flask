import numpy as np
import pandas as pd
import pytest

from app.models.lolopy_model import train_lolopy_model, evaluate_lolopy_model


def _small_active_learning_frame(labelled_count=5):
    rows = 12
    df = pd.DataFrame({
        "Idx_Sample": range(1, rows + 1),
        "water": np.linspace(0.3, 0.6, rows),
        "cement": np.linspace(300, 420, rows),
        "strength": [30 + i for i in range(labelled_count)] + [np.nan] * (rows - labelled_count),
    })
    return df


def test_lolopy_uses_early_cycle_fallback_below_native_minimum():
    data = _small_active_learning_frame(labelled_count=5)

    model, _, _ = train_lolopy_model(data, ["water", "cement"], ["strength"])
    results = evaluate_lolopy_model(
        model,
        data,
        ["water", "cement"],
        ["strength"],
        curiosity=0.5,
        weights_targets=np.array([1.0]),
        max_or_min_targets=["max"],
    )

    assert getattr(model, "model_backend") == "lolopy_early_cycle_rf_fallback"
    assert results.attrs["warnings"]
    assert "5 labelled rows" in results.attrs["warnings"][0]
    assert "Add 3 more labelled rows" in results.attrs["warnings"][0]
    assert not results.empty
    assert "Utility" in results.columns
    assert results["Selected for Testing"].any()


def test_lolopy_requires_two_labels_to_start():
    data = _small_active_learning_frame(labelled_count=1)

    with pytest.raises(ValueError, match="at least 2 labelled rows"):
        train_lolopy_model(data, ["water", "cement"], ["strength"])
