def accuracy(true_pos, false_pos, true_neg, false_neg):
    numerator = true_pos + true_neg
    denominator = true_pos + true_neg + false_pos + false_neg
    return numerator/denominator


def sensitivity(true_pos, false_neg):
    try:
        return true_pos/(true_pos + false_neg)
    except ZeroDivisionError:
        return float('nan')


def specificity(true_neg, false_pos):
    try:
        return true_neg/(true_neg + false_pos)

    except ZeroDivisionError:
        return float('nan')


def pos_pred_val(true_pos, false_pos):
    try:
        return true_pos/(true_pos + false_pos)
    except ZeroDivisionError:
        return float('nan')


def neg_pred_val(true_neg, false_neg):
    try:
        return true_neg/(true_neg + false_neg)
    except ZeroDivisionError:
        return float('nan')
