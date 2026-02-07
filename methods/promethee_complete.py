import pulp
import numpy as np
from methods.const import RELATION_SCORES


def promethee_complete_pulp(credibility, output=False, solver=None):
    n = len(credibility)

    cred = np.asarray(credibility, dtype=float)

    prob = pulp.LpProblem("ranking_model", pulp.LpMinimize)

    # Binary vars
    r = pulp.LpVariable.dicts("r", (range(n), range(n)), cat="Binary")
    z = pulp.LpVariable.dicts("z", (range(n), range(n)), cat="Binary")

    # R1, R2 + diagonal = 0
    for i in range(n):
        for k in range(n):
            if i != k:
                prob += r[i][k] + r[k][i] >= 1
                prob += r[i][k] + r[k][i] - 1 == z[i][k]
            else:
                prob += r[i][k] == 0
                prob += z[i][k] == 0

    # R4
    for i in range(n):
        for k in range(n):
            for p in range(n):
                if i != k and k != p and p != i:
                    prob += r[i][p] + r[p][k] - r[i][k] <= 1.5

    # Valued relations (constants)
    reversed_matrix = cred.T
    indifference_matrix = 1.0 - cred - reversed_matrix

    valued_relations = [
        {"var": cred, "rel": "PP"},
        {"var": reversed_matrix, "rel": "NP"},
        {"var": indifference_matrix, "rel": "I"},
    ]

    # Objective
    FN_terms = []
    for i in range(n):
        for k in range(i + 1, n):
            for s_relation in valued_relations:
                w = float(s_relation["var"][i][k])
                rel = s_relation["rel"]

                FN_terms.append(w * (r[i][k] - z[i][k]) * RELATION_SCORES[rel]["PP"])
                FN_terms.append(w * (r[k][i] - z[i][k]) * RELATION_SCORES[rel]["NP"])
                FN_terms.append(w * z[i][k] * RELATION_SCORES[rel]["I"])

    prob += pulp.lpSum(FN_terms)

    # Solve
    if solver is None:
        solver = pulp.PULP_CBC_CMD(msg=bool(output))
    prob.solve(solver)

    if prob.status != pulp.LpStatusOptimal:
        return None

    # Return matrices
    r_mat = np.zeros((n, n), dtype=int)
    z_mat = np.zeros((n, n), dtype=int)

    for i in range(n):
        for k in range(n):
            r_mat[i, k] = int(pulp.value(r[i][k]))
            z_mat[i, k] = int(pulp.value(z[i][k]))

    obj = pulp.value(prob.objective)
    return r_mat, z_mat, obj
