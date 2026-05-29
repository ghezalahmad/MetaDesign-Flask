import numpy as np
import pandas as pd
import pytest

from app.models.protonet_model import protonet_train, evaluate_protonet, ProtoNetModel


def _small_active_learning_frame(labelled_count=5):
    rows = 12
    return pd.DataFrame({
        "Idx_Sample": range(1, rows + 1),
        "water_binder_ratio": np.linspace(0.32, 0.48, rows),
        "slag_pct": np.linspace(20, 55, rows),
        "Fidelity_Level": [0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
        "fc_28d_MPa": [40 + i for i in range(labelled_count)] + [np.nan] * (rows - labelled_count),
    })


def test_protonet_uses_early_cycle_fallback_below_native_minimum():
    data = _small_active_learning_frame(labelled_count=5)
    model = ProtoNetModel(input_size=3, output_size=1)

    trained_model, _, _ = protonet_train(
        model,
        data,
        ["water_binder_ratio", "slag_pct", "Fidelity_Level"],
        ["fc_28d_MPa"],
    )
    results = evaluate_protonet(
        trained_model,
        data,
        ["water_binder_ratio", "slag_pct", "Fidelity_Level"],
        ["fc_28d_MPa"],
        curiosity=0.5,
        weights=np.array([1.0]),
        max_or_min=["max"],
    )

    assert getattr(trained_model, "model_backend") == "protonet_early_cycle_rf_fallback"
    assert results.attrs["warnings"]
    assert "5 labelled rows" in results.attrs["warnings"][0]
    assert "Add 5 more labelled rows" in results.attrs["warnings"][0]
    assert not results.empty
    assert "Utility" in results.columns
    assert results["Selected for Testing"].any()


def test_protonet_requires_two_labels_to_start():
    data = _small_active_learning_frame(labelled_count=1)
    model = ProtoNetModel(input_size=3, output_size=1)

    with pytest.raises(ValueError, match="at least 2 labelled rows"):
        protonet_train(
            model,
            data,
            ["water_binder_ratio", "slag_pct", "Fidelity_Level"],
            ["fc_28d_MPa"],
        )
