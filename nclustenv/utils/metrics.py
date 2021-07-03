
def IoU(x, y):

    """
    Returns the intersection over union for cluster `x`,  and cluster `y`.
    """

    intersection = sum([len(list(set(sx).intersection(sy))) for sx, sy in zip(x, y)])
    union = sum([(len(sx) + len(sy)) for sx, sy in zip(x, y)]) - intersection
    return float(intersection) / union


def match_score_1_n(fclust, hclusts):

    """
    For a given cluster `fclust` (found cluster), returns an IoU comparison against every element of `hclusts`
    (hidden clusters).
    """
    return [IoU(fclust, y) for y in hclusts]
