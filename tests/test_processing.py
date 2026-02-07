import json
from pathlib import Path
import numpy as np

from scripts.run_processing import process_files


def _make_valid_credibility(n: int) -> np.ndarray:
    m = np.random.rand(n, n)
    np.fill_diagonal(m, 0.0)
    return m


def test_process_files_creates_result_json(tmp_path: Path):
    inp = tmp_path / "input"
    out = tmp_path / "output"
    inp.mkdir(parents=True, exist_ok=True)

    n = 4
    data = {
        "crisp_matrix": _make_valid_credibility(n).tolist(),
        "electre_matrix": _make_valid_credibility(n).tolist(),
        "promethee_matrix": _make_valid_credibility(n).tolist(),
    }

    in_file = inp / "case_1.json"
    with open(in_file, "w", encoding="utf-8") as f:
        json.dump(data, f)

    process_files(str(inp), str(out), output=False)

    res_file = out / "result_case_1.json"
    assert res_file.exists()

    with open(res_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    # --- ensure base matrices are present
    for base_key in ["crisp_matrix", "electre_matrix", "promethee_matrix"]:
        assert base_key in results

    # --- expected result groups
    groups = [
        "crisp_complete_results",
        "crisp_partial_results",
        "electre_complete_results",
        "electre_partial_results",
        "promethee_complete_results",
        "promethee_partial_results",
    ]
    for g in groups:
        assert g in results

    def check_square(mat, n):
        assert isinstance(mat, list) and len(mat) == n
        assert all(isinstance(row, list) and len(row) == n for row in mat)

    def check_diag_zero(mat, n):
        assert all(mat[i][i] == 0 for i in range(n))

    def check_obj(obj):
        assert isinstance(obj, (int, float)), f"objective_value should be number, got {type(obj)}"

    # --- check complete example
    cc = results["crisp_complete_results"]
    assert "objective_value" in cc
    check_obj(cc["objective_value"])
    check_square(cc["r"], n); check_diag_zero(cc["r"], n)
    check_square(cc["z"], n); check_diag_zero(cc["z"], n)

    # --- check partial example
    pp = results["promethee_partial_results"]
    assert "objective_value" in pp
    check_obj(pp["objective_value"])
    for k in ["r", "P_plus", "P_minus", "I", "R"]:
        check_square(pp[k], n)
        check_diag_zero(pp[k], n)
