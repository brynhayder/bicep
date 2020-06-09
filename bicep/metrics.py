
# This doesn't class as a metric for ecaluate!
def n_correct(logits, target):
        pred = logits.argmax(dim=1, keepdim=True)
        # Need to call .item() before the division to python casts to float
        # Need to call .item() before the division to python casts to float
        # otherwise pytorch does integer division.
        # otherwise pytorch does integer division.
        return pred.eq(target.view_as(pred)).sum().item()


