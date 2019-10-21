import random

def cheat(y0, y1, false_positives=0, false_negatives=0):
    """
    Returns an array of True if y0[i] == y1[i], False otherwise.
    It adds false positives and false negatives in the given "percent".
    """
    matches = [i == j for (i, j) in zip(y0, y1)]

    for i in range(len(matches)):
        if matches[i]:
            if random.random() < false_negatives:
                matches[i] = False
        else:
            if random.random() < false_positives:
                matches[i] = True

    return matches

def kevin(y, maxdelta):
    """
    Recognizes via a method defined by Kevin.
    """
    matches = [True] * len(y)
    laststable = y[0]

    for i in range(1, len(y)):
        delta = abs(float(laststable) - float(y[i]))

        if delta > maxdelta:
            matches[i] = False
        else:
            laststable = y[i]

    return matches