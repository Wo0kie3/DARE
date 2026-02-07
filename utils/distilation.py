import numpy as np
from typing import List
import numpy as np
from typing import List

def s_func(x: float) -> float:
    return -0.15 * x + 0.3


def _validate_sigma(credibility: np.ndarray) -> np.ndarray:
    if not isinstance(credibility, np.ndarray):
        credibility = np.asarray(credibility, dtype=float)

    if credibility.ndim != 2 or credibility.shape[0] != credibility.shape[1]:
        raise ValueError(f"credibility must be a square (N×N) matrix, got shape={credibility.shape}")

    # Opcjonalnie: niech przekątna nie wpływa na lambda (chociaż i!=j ją omija)
    sigma = credibility.astype(float, copy=True)
    np.fill_diagonal(sigma, 0.0)
    return sigma


def _inner_distillation(indices: List[int], sigma: np.ndarray, lambda_: float, direction: int) -> List[int]:
    current = indices.copy()

    while True:
        next_lambda = 0.0
        for i in current:
            for j in current:
                if sigma[i, j] < lambda_ - s_func(lambda_):
                    next_lambda = max(next_lambda, sigma[i, j])

        if lambda_ == 0:
            return current

        lambda_ = next_lambda

        scores = {i: 0 for i in current}
        for i in current:
            for j in current:
                if i == j:
                    continue
                if sigma[i, j] > lambda_ and sigma[i, j] > sigma[j, i] + s_func(sigma[i, j]):
                    scores[i] += direction
                    scores[j] -= direction

        max_score = max(scores.values())
        best = [i for i, v in scores.items() if v == max_score]

        if len(best) == 1:
            return best

        current = best


def _distillation(sigma: np.ndarray, direction: int) -> np.ndarray:
    n = sigma.shape[0]
    levels = np.zeros(n, dtype=int)

    A = list(range(n))
    k = 1
    lambda_ = 0.0

    while True:
        for i in A:
            for j in A:
                if i != j and sigma[i, j] > lambda_:
                    lambda_ = sigma[i, j]

        next_lambda = 0.0
        for i in A:
            for j in A:
                if sigma[i, j] < lambda_ - s_func(lambda_):
                    next_lambda = max(next_lambda, sigma[i, j])

        if lambda_ == 0:
            for i in A:
                levels[i] = k * direction
            return levels

        lambda_ = next_lambda

        scores = {i: 0 for i in A}
        for i in A:
            for j in A:
                if i == j:
                    continue
                if sigma[i, j] > lambda_ and sigma[i, j] > sigma[j, i] + s_func(sigma[i, j]):
                    scores[i] += direction
                    scores[j] -= direction

        max_score = max(scores.values())
        best = [i for i, v in scores.items() if v == max_score]

        if len(best) != 1:
            best = _inner_distillation(best, sigma, lambda_, direction)

        for i in best:
            levels[i] = k * direction

        A = [i for i in A if levels[i] == 0]
        if not A:
            return levels

        k += 1


def descending_distillation(credibility: np.ndarray) -> np.ndarray:
    sigma = _validate_sigma(credibility)
    return _distillation(sigma, direction=1)


def ascending_distillation(credibility: np.ndarray) -> np.ndarray:
    sigma = _validate_sigma(credibility)
    return _distillation(sigma, direction=-1)


def asc_to_desc_format(asc_levels: np.ndarray) -> np.ndarray:
    """
    Zamienia ranking ascending (ujemny, mniejsze = lepsze)
    na format descending (dodatni, mniejsze = lepsze),
    zachowując remisy.

    Przykład:
      asc  = [-3, -1, -3, -2]
      desc = [ 1,  3,  1,  2]
    """
    asc = np.asarray(asc_levels, dtype=int)

    # unikalne poziomy, posortowane od najlepszego do najgorszego
    unique_sorted = np.unique(asc)
    unique_sorted.sort()  # mniejsze (bardziej ujemne) = lepsze

    mapping = {val: rank + 1 for rank, val in enumerate(unique_sorted)}

    return np.array([mapping[v] for v in asc], dtype=int)



UNCOMPARABLE = 0
PREFERRED = 1
WORSE = 2
INDIFFERENT = 3


def _preorder_cmp(value_i: int, value_j: int) -> int:
    """
    Porównanie w preorderze wg zasady:
      mniejsza liczba => lepsza pozycja

    Zwraca:
      PREFERRED jeśli i lepsze od j
      WORSE jeśli i gorsze od j
      INDIFFERENT jeśli remis
    """
    if value_i < value_j:
        return PREFERRED
    if value_i > value_j:
        return WORSE
    return INDIFFERENT


def final_preorder_intersection(desc_levels: np.ndarray, asc_levels: np.ndarray) -> np.ndarray:
    """
    Final preorder = przecięcie preorderów descending i ascending.

    desc_levels: >0, mniejsze lepsze
    asc_levels : <0, mniejsze (bardziej ujemne) lepsze

    Reguły (zgodne z opisem):
      - i ≻ j jeśli: i nie jest gorsze w żadnym preorderze i jest ściśle lepsze w co najmniej jednym
      - i ~ j jeśli: remis w obu
      - i || j jeśli: w jednym i ≻ j, a w drugim j ≻ i
    """
    desc_levels = np.asarray(desc_levels, dtype=int)
    asc_levels = np.asarray(asc_levels, dtype=int)

    if desc_levels.ndim != 1 or asc_levels.ndim != 1 or desc_levels.shape != asc_levels.shape:
        raise ValueError("desc_levels and asc_levels must be 1D arrays with the same shape")

    n = desc_levels.shape[0]
    rel = np.zeros((n, n), dtype=int)

    for i in range(n):
        rel[i, i] = INDIFFERENT
        for j in range(i + 1, n):
            d = _preorder_cmp(desc_levels[i], desc_levels[j])
            a = _preorder_cmp(asc_levels[i], asc_levels[j])

            if d == INDIFFERENT and a == INDIFFERENT:
                rel[i, j] = INDIFFERENT
                rel[j, i] = INDIFFERENT

            elif (d == PREFERRED and a == WORSE) or (d == WORSE and a == PREFERRED):
                rel[i, j] = UNCOMPARABLE
                rel[j, i] = UNCOMPARABLE

            elif (d != WORSE) and (a != WORSE) and (d == PREFERRED or a == PREFERRED):
                rel[i, j] = PREFERRED
                rel[j, i] = WORSE

            elif (d != PREFERRED) and (a != PREFERRED) and (d == WORSE or a == WORSE):
                rel[i, j] = WORSE
                rel[j, i] = PREFERRED

            else:
                # powinno się nie zdarzać przy spójnej definicji,
                # ale zostawiam jako bezpieczny fallback
                rel[i, j] = UNCOMPARABLE
                rel[j, i] = UNCOMPARABLE

    return rel


def ranks_longest_path_to_top(relations: np.ndarray) -> np.ndarray:
    """
    Rank = długość najdłuższej ścieżki do jakiejś TOP + 1.

    Budujemy graf skierowany "gorszy -> lepszy":
      jeśli i ≻ j (i preferred j) to krawędź: j -> i

    TOP: wierzchołki bez wyjść (nikt nie jest od nich lepszy).

    Zwraca ranks: 1 = top.
    Wymaga, żeby graf preferencji był acykliczny; jeśli jest cykl => błąd.
    """
    rel = np.asarray(relations, dtype=int)
    n = rel.shape[0]
    if rel.shape != (n, n):
        raise ValueError("relations must be NxN")

    # worse -> better adjacency
    adj = [[] for _ in range(n)]
    indeg = np.zeros(n, dtype=int)  # dla grafu better->worse (patrz niżej)

    # zbuduj rev: better -> worse oraz indeg[worse]
    rev = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if rel[i, j] == PREFERRED:
                # i ≻ j => edge j -> i (worse -> better)
                adj[j].append(i)
                # odwrócony: i -> j (better -> worse)
                rev[i].append(j)
                indeg[j] += 1

    # topologiczne sortowanie na grafie better->worse
    queue = [v for v in range(n) if indeg[v] == 0]  # TOP-y mają indeg 0
    order: List[int] = []

    while queue:
        x = queue.pop()
        order.append(x)
        for w in rev[x]:
            indeg[w] -= 1
            if indeg[w] == 0:
                queue.append(w)

    if len(order) != n:
        raise ValueError("Cycle detected in strict preference graph (final preorder).")

    # dist_from_top[w] = max długość ścieżki TOP -> w (better->worse)
    dist = np.zeros(n, dtype=int)
    for x in order:
        for w in rev[x]:
            dist[w] = max(dist[w], dist[x] + 1)

    return dist + 1

def create_preference_matrix_low_high(ranking: np.ndarray) -> np.ndarray:
    """
    Buduje macierz preferencji (preorder),
    zakładając że:
      - im niższa wartość w rankingu, tym lepsza pozycja

    matrix[i, j] = 1  ⇔  i jest co najmniej tak dobre jak j
    """
    ranking = np.asarray(ranking)
    N = ranking.shape[0]

    matrix = np.zeros((N, N), dtype=int)

    for i in range(N):
        for j in range(N):
            if ranking[i] < ranking[j] or ranking[i] == ranking[j]:
                matrix[i, j] = 1

    return matrix

def tie_groups_from_matrix(M: np.ndarray):
    """
    Grupuje alternatywy w klasy remisowe na podstawie relacji:
    remis(i,j) <=> M[i,j]==1 and M[j,i]==1
    (spójne składowe w grafie remisów).
    """
    M = np.asarray(M)
    n = M.shape[0]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # łączymy tylko pary w remisie (wzajemne 1)
    for i in range(n):
        for j in range(i + 1, n):
            if M[i, j] == 1 and M[j, i] == 1:
                union(i, j)

    groups = {}
    for i in range(n):
        r = find(i)
        groups.setdefault(r, []).append(i)

    return list(groups.values())

def break_ties_with_asc_desc(M, asc, desc):
    M = np.asarray(M)
    asc = np.asarray(asc)
    desc = np.asarray(desc)
    n = len(asc)

    groups = tie_groups_from_matrix(M)

    # sortowanie wewnątrz każdej grupy remisowej po (desc, asc, index)
    ordered = []
    for g in groups:
        g_sorted = sorted(g, key=lambda i: (desc[i], asc[i], i))
        ordered.append(g_sorted)

    # żeby mieć "globalną" kolejność, układam grupy wg najlepszego elementu w grupie
    # (też tylko na podstawie asc/desc, bez dodatkowych reguł):
    ordered.sort(key=lambda g: (desc[g[0]], asc[g[0]], g[0]))

    final_order = [i for g in ordered for i in g]

    # rangi 1..n
    final_rank = np.empty(n, dtype=int)
    for r, i in enumerate(final_order, start=1):
        final_rank[i] = r

    return final_order, final_rank, groups

def final_rank(credibility):
    desc = descending_distillation(credibility)
    asc = ascending_distillation(credibility)
    asc = asc_to_desc_format(asc)
    final_rel = final_preorder_intersection(desc, asc)
    ranks = ranks_longest_path_to_top(final_rel)
    return create_preference_matrix_low_high(ranks)

def median_rank(credibility):
    desc = descending_distillation(credibility)
    asc = ascending_distillation(credibility)
    asc = asc_to_desc_format(asc)
    final_rel = final_preorder_intersection(desc, asc)
    ranks = ranks_longest_path_to_top(final_rel)
    ranks = create_preference_matrix_low_high(ranks)
    _, rank, _ = break_ties_with_asc_desc(ranks, asc, desc)
    return create_preference_matrix_low_high(rank)
