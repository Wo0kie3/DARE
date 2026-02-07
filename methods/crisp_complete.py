import pulp
import numpy as np
from methods.util import RELATION_SCORES, get_relation


def crisp_complete_pulp(credibility, output=False, solver=None):
    n = len(credibility)

    # Problem
    prob = pulp.LpProblem("crisp_complete", pulp.LpMinimize)

    # Binary decision variables
    r = pulp.LpVariable.dicts("r", (range(n), range(n)), cat="Binary")
    z = pulp.LpVariable.dicts("z", (range(n), range(n)), cat="Binary")

    # R1, R2
    for i in range(n):
        for k in range(n):
            if i != k:
                prob += r[i][k] + r[k][i] >= 1
                prob += r[i][k] + r[k][i] - 1 == z[i][k]
            else:
                # enforce diagonal = 0 explicitly
                prob += r[i][k] == 0
                prob += z[i][k] == 0

    # R4
    for i in range(n):
        for k in range(n):
            for p in range(n):
                if i != k and k != p and p != i:
                    prob += r[i][p] + r[p][k] - r[i][k] <= 1.5

    # Objective
    FN = []
    for i in range(n):
        for k in range(i + 1, n):
            rel = get_relation(credibility[i][k], credibility[k][i])
            s = RELATION_SCORES[rel]

            FN.append(r[i][k] * s["PP"])
            FN.append(r[k][i] * s["NP"])
            FN.append(z[i][k] * (s["I"] - s["PP"] - s["NP"]))

    prob += pulp.lpSum(FN)

    # Solve
    if solver is None:
        solver = pulp.PULP_CBC_CMD(msg=bool(output))
    prob.solve(solver)

    if prob.status != pulp.LpStatusOptimal:
        return None

    # Build matrices (0/1, diagonal = 0)
    r_mat = np.zeros((n, n), dtype=int)
    z_mat = np.zeros((n, n), dtype=int)

    for i in range(n):
        for k in range(n):
            r_mat[i, k] = int(pulp.value(r[i][k]))
            z_mat[i, k] = int(pulp.value(z[i][k]))

    obj = pulp.value(prob.objective)
    return r_mat, z_mat, obj
