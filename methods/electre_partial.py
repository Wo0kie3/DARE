import pulp
import numpy as np
from methods.const import RELATION_SCORES


def electre_partial_pulp(credibility, output=False, solver=None):
    n = len(credibility)

    # ensure numpy for .T and vector ops
    cred = np.asarray(credibility, dtype=float)

    prob = pulp.LpProblem("ranking_model", pulp.LpMinimize)

    # Binary decision variables
    r = pulp.LpVariable.dicts("r", (range(n), range(n)), cat="Binary")
    P_plus = pulp.LpVariable.dicts("P_plus", (range(n), range(n)), cat="Binary")
    P_minus = pulp.LpVariable.dicts("P_minus", (range(n), range(n)), cat="Binary")
    I = pulp.LpVariable.dicts("I", (range(n), range(n)), cat="Binary")
    R = pulp.LpVariable.dicts("R", (range(n), range(n)), cat="Binary")

    # Constraints (R1..R5) + diagonal fixed to 0
    for i in range(n):
        for k in range(n):
            if i != k:
                prob += r[i][k] - r[k][i] - P_plus[i][k] <= 0   # R1
                prob += r[k][i] - r[i][k] - P_minus[i][k] <= 0  # R2
                prob += r[i][k] + r[k][i] - I[i][k] <= 1        # R3
                prob += 1 - r[i][k] - r[k][i] - R[i][k] <= 0    # R4
                prob += P_plus[i][k] + P_minus[i][k] + I[i][k] + R[i][k] == 1  # R5
            else:
                # diagonal = 0
                prob += r[i][k] == 0
                prob += P_plus[i][k] == 0
                prob += P_minus[i][k] == 0
                prob += I[i][k] == 0
                prob += R[i][k] == 0

    # R7
    for i in range(n):
        for k in range(n):
            for p in range(n):
                if i != k and k != p and p != i:
                    prob += r[i][p] + r[p][k] - r[i][k] <= 1.5

    # Valued relations (constants)
    reversed_matrix = cred.T
    positive_preference_matrix = np.minimum(cred, 1.0 - reversed_matrix)
    negative_preference_matrix = np.minimum(1.0 - cred, reversed_matrix)
    indifference_matrix = np.minimum(cred, reversed_matrix)
    incomparible_matrix = np.minimum(1.0 - cred, 1.0 - reversed_matrix)

    problem_relations = [
        {"var": P_plus, "rel": "PP"},
        {"var": P_minus, "rel": "NP"},
        {"var": I, "rel": "I"},
        {"var": R, "rel": "R"},
    ]
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
            for p_rel in problem_relations:
                for s_rel in valued_relations:
                    w = float(s_rel["var"][i][k])          # constant weight
                    s_key = s_rel["rel"]                   # valued relation key
                    p_key = p_rel["rel"]                   # chosen crisp relation key
                    score = RELATION_SCORES[s_key][p_key]  # constant score
                    FN_terms.append(w * p_rel["var"][i][k] * score)

    prob += pulp.lpSum(FN_terms)

    # Solve
    if solver is None:
        solver = pulp.PULP_CBC_CMD(msg=bool(output))
    prob.solve(solver)

    if prob.status != pulp.LpStatusOptimal:
        return None

    # Return matrices
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
