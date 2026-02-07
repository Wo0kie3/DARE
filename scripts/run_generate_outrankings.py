import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.helper_functions import net_flow_score_pos_neg, net_flow_score, ranking_to_weak_preference_matrix
from utils.distilation import final_rank, median_rank

# ===== mcda imports (dopasuj jeśli masz inne ścieżki / nazwy) =====
from mcda.scales import QuantitativeScale, PreferenceDirection
from mcda.core.performance_table import PerformanceTable
from mcda.outranking.electre import Electre3
from mcda.outranking.promethee import Promethee1, Promethee2
from mcda.outranking.promethee.preference_functions import VShapeFunction
from mcda.outranking.preference_structure import PreferenceStructure

def generate_outrankings(input_dir, output_dir):
    """
    Generate outranking results based on ELECTRE III, PROMETHEE I/II, and crisp outrankings.

    Parameters:
    - input_dir: str|Path - Directory containing input CSV files.
    - output_dir: str|Path - Directory to save the outranking results.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # collect all base CSVs (without _weights)
    csv_files = []
    for p in input_dir.rglob("*.csv"):
        if p.name.endswith("_weights.csv"):
            continue
        csv_files.append(p)

    for file_path in tqdm(csv_files, desc="Generating outrankings", unit="file"):
        weights_file = file_path.with_name(file_path.stem + "_weights.csv")
        if not weights_file.exists():
            raise FileNotFoundError(f"Missing weights file for {file_path}: {weights_file}")

        # Load performances and weights
        data = pd.read_csv(file_path)
        weights_data = pd.read_csv(weights_file)

        performances = data.values
        weights = weights_data["Weight"].values

        alt_len = performances.shape[0]
        crit_len = len(weights)

        optimization = weights_data["Optimization"].values.tolist()
        q_vals = weights_data["Q"].values
        p_vals = weights_data["P"].values
        v_vals = weights_data["V"].values
        cutoff = float(weights_data["cutoff"].iloc[0])

        # Build MCDA structures
        scales = {}
        preference_func_list = {}
        W, I, P, V = {}, {}, {}, {}

        for i in range(crit_len):
            pref_dir = PreferenceDirection.MAX if optimization[i] == "max" else PreferenceDirection.MIN
            scales[i] = QuantitativeScale(0, 1, preference_direction=pref_dir)

            W[i] = float(weights[i])
            I[i] = float(q_vals[i])
            P[i] = float(p_vals[i])
            V[i] = float(v_vals[i])

            preference_func_list[i] = VShapeFunction(p=float(p_vals[i]), q=float(q_vals[i]))

        dataset = PerformanceTable(
            performances,
            alternatives=[str(i) for i in range(alt_len)],
            scales=scales
        )

        promethee1 = Promethee1(dataset, W, preference_func_list)
        promethee2 = Promethee2(dataset, W, preference_func_list)
        electre3 = Electre3(
            performance_table=dataset,
            criteria_weights=W,
            indifference_thresholds=I,
            preference_thresholds=P,
            veto_thresholds=V
        )

        # Matrices
        promethee_matrix = promethee1.preferences(promethee1.partial_preferences()).data
        electre_matrix = electre3.construct().data
        crisp_matrix = (electre_matrix >= cutoff).astype(int)

        # Rankings/matrices from rankings
        promethee1_rank = promethee1.rank().outranking_matrix.data.values
        promethee2_rank = PreferenceStructure().from_ranking(promethee2.rank()).outranking_matrix.data.values

        # === your custom ranking helpers ===
        crisp_nfs = net_flow_score(crisp_matrix)
        crisp_nfs_pos_neg = ranking_to_weak_preference_matrix(net_flow_score_pos_neg(crisp_matrix))

        # ELECTRE distillations
        electre3_median_rank = median_rank(electre_matrix)
        electre3_final_ranking = final_rank(electre_matrix)

        # Results dict (JSON serializable)
        results = {
            "promethee_matrix": promethee_matrix.values.tolist(),
            "electre_matrix": electre_matrix.values.tolist(),
            "crisp_matrix": crisp_matrix.values.tolist(),
            "promethee1_rank": promethee1_rank.tolist(),
            "promethee2_rank": promethee2_rank.tolist(),
            "crisp_nfs": crisp_nfs.tolist(),
            "crisp_nfs_pos_neg": crisp_nfs_pos_neg.tolist(),
            "electre3_median_rank": electre3_median_rank.tolist(),
            "electre3_final_ranking": electre3_final_ranking.tolist(),
            "meta": {
                "cutoff": cutoff,
                "source_csv": str(file_path),
                "source_weights_csv": str(weights_file),
            }
        }

        # Preserve directory structure under output_dir
        rel_dir = file_path.parent.relative_to(input_dir)
        out_dir = output_dir / rel_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        results_path = out_dir / f"{file_path.stem}_results.json"
        with results_path.open("w", encoding="utf-8") as outfile:
            json.dump(results, outfile, indent=4)


if __name__ == "__main__":
    outranking_parameters = {
        "input_dir": "data",
        "output_dir": "outrankings"
    }

    generate_outrankings(
        input_dir=outranking_parameters["input_dir"],
        output_dir=outranking_parameters["output_dir"]
    )
