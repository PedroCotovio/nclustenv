
def IoU(x, y):

    intersection = sum([len(list(set(sx).intersection(sy))) for sx, sy in zip(x, y)])
    union = sum([(len(sx) + len(sy)) for sx, sy in zip(x, y)]) - intersection
    return float(intersection) / union


def match_score_1_n(fbics, hbicsq):
    return [IoU(fbics, y) for y in hbicsq]
