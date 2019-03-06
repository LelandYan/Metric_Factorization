import math
import numpy as np

# efficient version
def precision_recall(k, rankedlist, test_matrix):
    b1 = rankedlist
    b2 = test_matrix
    s2 = set(b2)
    hits = [ (idx, val) for idx, val in enumerate(b1) if val in s2]
    count = len(hits)
    return float(count / k), float(count / len(test_matrix))

