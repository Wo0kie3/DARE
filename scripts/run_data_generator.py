import random
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Union


def oblicz_wagi(num_criteria: int, alpha: float = 1.0) -> np.ndarray:
    """
    Losuje wagi sumujące się do 1 (Dirichlet).
    alpha=1.0 -> równomiernie po simpleksie,
    większe alpha -> bardziej wyrównane wagi.
    """
    return np.random.dirichlet([alpha] * num_criteria)


def generate_data(
    criteria_range: tuple[int, int],
    alternatives_range: tuple[int, int],
    thresholds: dict[str, tuple[float, float, float]],
    output_dir: Union[str, Path],
    n_files_per_setting: int = 100,
    cutoffs: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8),
    seed: Union[int, None] = None,
) -> None:
    """
    Generate experimental data based on given parameters and save as:
    - data_i.csv
    - data_i_weights.csv

    Directory structure:
    output_dir/
      criteria_{c}/alternatives_{a}/level_{level}/cutoff_{cutoff}/
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Progress total
    total = 0
    for num_criteria in range(criteria_range[0], criteria_range[1] + 1):
        for num_alts in range(alternatives_range[0], alternatives_range[1] + 1):
            for _level in thresholds.keys():
                for _cutoff in cutoffs:
                    total += n_files_per_setting

    pbar = tqdm(total=total, desc="Generating data", unit="file")

    for num_criteria in range(criteria_range[0], criteria_range[1] + 1):
        criteria_dir = output_dir / f"criteria_{num_criteria}"
        criteria_dir.mkdir(parents=True, exist_ok=True)

        for num_alternatives in range(alternatives_range[0], alternatives_range[1] + 1):
            alternatives_dir = criteria_dir / f"alternatives_{num_alternatives}"
            alternatives_dir.mkdir(parents=True, exist_ok=True)

            for level, (q, p, v) in thresholds.items():
                level_dir = alternatives_dir / f"level_{level}"
                level_dir.mkdir(parents=True, exist_ok=True)

                for cutoff in cutoffs:
                    cutoff_dir = level_dir / f"cutoff_{cutoff}"
                    cutoff_dir.mkdir(parents=True, exist_ok=True)

                    for i in range(n_files_per_setting):
                        file_path = cutoff_dir / f"data_{i+1}.csv"

                        # Weights
                        weights = oblicz_wagi(num_criteria)

                        # Performances
                        performances = np.random.uniform(0, 1, size=(num_alternatives, num_criteria))

                        # Optimization per criterion
                        optimization = ["max" if random.random() > 0.5 else "min" for _ in range(num_criteria)]

                        # Lambda
                        lambda_value = random.uniform(0.5, 1.0)

                        # Save performance CSV
                        df = pd.DataFrame(
                            performances,
                            columns=[f"Criterion_{c+1}" for c in range(num_criteria)]
                        )
                        df.to_csv(file_path, index=False)

                        # Save weights/meta CSV
                        weights_file = file_path.with_name(file_path.stem + "_weights.csv")
                        weights_df = pd.DataFrame({
                            "Criterion": [f"Criterion_{c+1}" for c in range(num_criteria)],
                            "Weight": weights,
                            "Optimization": optimization,
                            "Lambda": [lambda_value - float(np.min(weights))] * num_criteria,
                            "Q": [q] * num_criteria,
                            "P": [p] * num_criteria,
                            "V": [v] * num_criteria,
                            "cutoff": [cutoff] * num_criteria,
                        })
                        weights_df.to_csv(weights_file, index=False)

                        pbar.update(1)

    pbar.close()


if __name__ == "__main__":
    data_parameters = {
        "criteria_range": (3, 8),
        "alternatives_range": (4, 10),
        "thresholds": {
            "low": (0.05, 0.15, 0.25),
            "medium": (0.15, 0.3, 0.5),
            "high": (0.25, 0.45, 0.75),
        },
        "output_dir": "new_data",
    }

    generate_data(
        criteria_range=data_parameters["criteria_range"],
        alternatives_range=data_parameters["alternatives_range"],
        thresholds=data_parameters["thresholds"],
        output_dir=data_parameters["output_dir"],
        n_files_per_setting=100,
        cutoffs=(0.5, 0.6, 0.7, 0.8),
        seed=67
    )
