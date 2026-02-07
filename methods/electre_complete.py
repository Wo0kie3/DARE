import pulp
import numpy as np
from methods.util import RELATION_SCORES


def electre_complete_pulp(credibility, output=False, solver=None):
    n = len(credibility)

    # Ensure numpy array (for .T and indexing)
    cred = np.asarray(credibility, dtype=float)

    prob = pulp.LpProblem("ranking_model", pulp.LpMinimize)

    # Binary vars
    r = pulp.LpVariable.dicts("r", (range(n), range(n)), cat="Binary")
    z = pulp.LpVariable.dicts("z", (range(n), range(n)), cat="Binary")

    # Constraints (R1, R2) + diagonal = 0
    for i in range(n):
        for k in range(n):
            if i != k:
                prob += r[i][k] + r[k][i] >= 1
                prob += r[i][k] + r[k][i] - 1 == z[i][k]
            else:
                prob += r[i][k] == 0
                prob += z[i][k] == 0

    # Constraint (R4)
    for i in range(n):
        for k in range(n):
            for p in range(n):
                if i != k and k != p and p != i:
                    prob += r[i][p] + r[p][k] - r[i][k] <= 1.5

    # Valued relations matrices (constants)
    reversed_matrix = cred.T
    positive_preference_matrix = np.minimum(cred, 1.0 - reversed_matrix)
    negative_preference_matrix = np.minimum(1.0 - cred, reversed_matrix)
    indifference_matrix = np.minimum(cred, reversed_matrix)
    incomparible_matrix = np.minimum(1.0 - cred, 1.0 - reversed_matrix)

    valued_relations = [
        {"var": positive_preference_matrix, "rel": "PP"},
        {"var": negative_preference_matrix, "rel": "NP"},
        {"var": indifference_matrix, "rel": "I"},
        {"var": incomparible_matrix, "rel": "R"},
    ]

    # Objective
    FN_terms = []
    for i in range(n):
        for k in range(i + 1, n):
            for s_relation in valued_relations:
                w = float(s_relation["var"][i][k])
                rel = s_relation["rel"]

                # exactly like your Gurobi expression:
                # w * (r[i,k] - z[i,k]) * score[rel]['PP']
                # + (r[k,i] - z[i,k]) * w * score[rel]['NP']
                # + z[i,k] * w * score[rel]['I']
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

    # Return matrices (0/1)
    r_mat = np.zeros((n, n), dtype=int)
    z_mat = np.zeros((n, n), dtype=int)

    for i in range(n):
        for k in range(n):
            r_mat[i, k] = int(pulp.value(r[i][k]))
            z_mat[i, k] = int(pulp.value(z[i][k]))

    obj = pulp.value(prob.objective)
    return r_mat, z_mat, obj
