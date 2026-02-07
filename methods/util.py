RELATION_SCORES = {
    "PP": {
        "PP": 0,
        "NP": 4,
        "I": 2,
        "R": 3
    },
    "NP": {
        "PP": 4,
        "NP": 0,
        "I": 2,
        "R": 3
    },
    "I": {
        "PP": 2,
        "NP": 2,
        "I": 0,
        "R": 2
    },
    "R": {
        "PP": 3,
        "NP": 3,
        "I": 2,
        "R": 0
    }
}

def get_relation(a, b):
    if a > b:
        return 'PP'
    elif a < b:
        return 'NP'
    elif a == b == 1:
        return 'I'
    else:
        return 'R'
    
def get_relation_score(rel_a, rel_b, score=RELATION_SCORES):
    return score[rel_a][rel_b]