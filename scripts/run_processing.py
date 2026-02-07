import os
import json
import numpy as np
from tqdm import tqdm

from methods.crisp_complete import crisp_complete_pulp
from methods.crisp_partial import crisp_partial_pulp
from methods.electre_complete import electre_complete_pulp
from methods.electre_partial import electre_partial_pulp
from methods.promethee_complete import promethee_complete_pulp
from methods.promethee_partial import promethee_partial_pulp


def _to_np_matrix(x):
    return np.asarray(x, dtype=float)


def _mat_to_list(mat):
    return mat.tolist()


def process_files(input_directory, output_directory, output=False):
    # --- 1) collect all json files first (to know total)
    json_files = []
    for subdir, _, files in os.walk(input_directory):
        for file in files:
            if file.endswith(".json"):
                json_files.append((subdir, file))

    # --- 2) process with progress bar
    for subdir, file in tqdm(json_files, desc="Processing files", unit="file"):
        file_path = os.path.join(subdir, file)

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        crisp_matrix = _to_np_matrix(data["crisp_matrix"])
        electre_matrix = _to_np_matrix(data["electre_matrix"])
        promethee_matrix = _to_np_matrix(data["promethee_matrix"])

        # Run models (with objective)
        r_cc, z_cc, obj_cc = crisp_complete_pulp(crisp_matrix, output=output)
        r_cp, Pp_cp, Pm_cp, I_cp, R_cp, obj_cp = crisp_partial_pulp(crisp_matrix, output=output)

        r_ec, z_ec, obj_ec = electre_complete_pulp(electre_matrix, output=output)
        r_ep, Pp_ep, Pm_ep, I_ep, R_ep, obj_ep = electre_partial_pulp(electre_matrix, output=output)

        r_pc, z_pc, obj_pc = promethee_complete_pulp(promethee_matrix, output=output)
        r_pp, Pp_pp, Pm_pp, I_pp, R_pp, obj_pp = promethee_partial_pulp(promethee_matrix, output=output)

        # Pack results
        results = {
            "crisp_matrix": _mat_to_list(crisp_matrix),
            "electre_matrix": _mat_to_list(electre_matrix),
            "promethee_matrix": _mat_to_list(promethee_matrix),

            "crisp_complete_results": {
                "objective_value": float(obj_cc),
                "r": _mat_to_list(r_cc),
                "z": _mat_to_list(z_cc),
            },
            "crisp_partial_results": {
                "objective_value": float(obj_cp),
                "r": _mat_to_list(r_cp),
                "P_plus": _mat_to_list(Pp_cp),
                "P_minus": _mat_to_list(Pm_cp),
                "I": _mat_to_list(I_cp),
                "R": _mat_to_list(R_cp),
            },

            "electre_complete_results": {
                "objective_value": float(obj_ec),
                "r": _mat_to_list(r_ec),
                "z": _mat_to_list(z_ec),
            },
            "electre_partial_results": {
                "objective_value": float(obj_ep),
                "r": _mat_to_list(r_ep),
                "P_plus": _mat_to_list(Pp_ep),
                "P_minus": _mat_to_list(Pm_ep),
                "I": _mat_to_list(I_ep),
                "R": _mat_to_list(R_ep),
            },

            "promethee_complete_results": {
                "objective_value": float(obj_pc),
                "r": _mat_to_list(r_pc),
                "z": _mat_to_list(z_pc),
            },
            "promethee_partial_results": {
                "objective_value": float(obj_pp),
                "r": _mat_to_list(r_pp),
                "P_plus": _mat_to_list(Pp_pp),
                "P_minus": _mat_to_list(Pm_pp),
                "I": _mat_to_list(I_pp),
                "R": _mat_to_list(R_pp),
            },
        }

        # Output path (keep structure)
        output_subdir = subdir.replace(input_directory, output_directory)
        os.makedirs(output_subdir, exist_ok=True)

        result_file_path = os.path.join(output_subdir, f"result_{file}")
        with open(result_file_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
