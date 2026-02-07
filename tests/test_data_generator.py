from pathlib import Path
import pandas as pd

from scripts.run_data_generator import generate_data


def test_generate_data_creates_files(tmp_path: Path):
    out_dir = tmp_path / "out"

    params = {
        "criteria_range": (3, 3),
        "alternatives_range": (4, 4),
        "thresholds": {"low": (0.05, 0.15, 0.25)},
    }

    generate_data(
        criteria_range=params["criteria_range"],
        alternatives_range=params["alternatives_range"],
        thresholds=params["thresholds"],
        output_dir=out_dir,
        n_files_per_setting=2,
        cutoffs=(0.5,),
        seed=123,
    )

    target_dir = out_dir / "criteria_3" / "alternatives_4" / "level_low" / "cutoff_0.5"
    assert target_dir.exists()

    csv1 = target_dir / "data_1.csv"
    w1 = target_dir / "data_1_weights.csv"
    assert csv1.exists()
    assert w1.exists()

    df = pd.read_csv(csv1)
    assert df.shape == (4, 3)

    wdf = pd.read_csv(w1)
    assert set(["Criterion", "Weight", "Optimization", "Lambda", "Q", "P", "V", "cutoff"]).issubset(wdf.columns)
    assert len(wdf) == 3
    assert abs(wdf["Weight"].sum() - 1.0) < 1e-6
