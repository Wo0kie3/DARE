import pulp
import numpy as np
from methods.util import RELATION_SCORES, get_relation


def crisp_partial_pulp(credibility, output=False, solver=None):
    n = len(credibility)

    prob = pulp.LpProblem("crisp_partial", pulp.LpMinimize)

    # Binary decision variables
    r = pulp.LpVariable.dicts("r", (range(n), range(n)), cat="Binary")
    P_plus = pulp.LpVariable.dicts("P_plus", (range(n), range(n)), cat="Binary")
    P_minus = pulp.LpVariable.dicts("P_minus", (range(n), range(n)), cat="Binary")
    I = pulp.LpVariable.dicts("I", (range(n), range(n)), cat="Binary")
    R = pulp.LpVariable.dicts("R", (range(n), range(n)), cat="Binary")

    # Constraints (R1..R5) for i != k, and diagonal fixed to 0
    for i in range(n):
        for k in range(n):
            if i != k:
                # R1..R5 (same as Gurobi)
                prob += r[i][k] - r[k][i] - P_plus[i][k] <= 0
                prob += r[k][i] - r[i][k] - P_minus[i][k] <= 0
                prob += r[i][k] + r[k][i] - I[i][k] <= 1
                prob += 1 - r[i][k] - r[k][i] - R[i][k] <= 0
                prob += P_plus[i][k] + P_minus[i][k] + I[i][k] + R[i][k] == 1
            else:
                # diagonal = 0
                prob += r[i][k] == 0
                prob += P_plus[i][k] == 0
                prob += P_minus[i][k] == 0
                prob += I[i][k] == 0
                prob += R[i][k] == 0

    # Transitivity-like constraint (R7)
    for i in range(n):
        for k in range(n):
            for p in range(n):
                if i != k and k != p and p != i:
                    prob += r[i][p] + r[p][k] - r[i][k] <= 1.5

    # Objective
    FN_terms = []
    for i in range(n):
        for k in range(i + 1, n):
            rel = get_relation(credibility[i][k], credibility[k][i])
            s = RELATION_SCORES[rel]
            FN_terms.append(P_plus[i][k] * s["PP"])
            FN_terms.append(P_minus[i][k] * s["NP"])
            FN_terms.append(I[i][k] * s["I"])
            FN_terms.append(R[i][k] * s["R"])

    prob += pulp.lpSum(FN_terms)

    # Solve
    if solver is None:
        solver = pulp.PULP_CBC_CMD(msg=bool(output))
    prob.solve(solver)

    if prob.status != pulp.LpStatusOptimal:
        return None

    # Return matrices (0/1)
    def to_mat(var_dict):
        mat = np.zeros((n, n), dtype=int)
        for i in range(n):
            for k in range(n):
                mat[i, k] = int(pulp.value(var_dict[i][k]))
        return mat

    r_mat = to_mat(r)
    P_plus_mat = to_mat(P_plus)
    P_minus_mat = to_mat(P_minus)
    I_mat = to_mat(I)
    R_mat = to_mat(R)

    obj = pulp.value(prob.objective)
    return r_mat, P_plus_mat, P_minus_mat, I_mat, R_mat, obj
