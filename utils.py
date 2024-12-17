class AverageMeter(object):
    def __init__(self):
        self.clear()

    def reset(self):
        self.avg = 0
        self.val = 0
        self.sum = 0
        self.count = 0

    def clear(self):
        self.reset()
        self.history = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count
        else:
            self.avg = 'nan'

    def new_epoch(self):
        self.history.append(self.avg)
        self.reset()


def get_iou(predictions, gt):
    nsamples, nclasses, height, width = predictions.size()
    prediction_max, prediction_argmax = predictions.max(-3)
    prediction_argmax = prediction_argmax.long()
    classes = gt.new_tensor([c for c in range(nclasses)]).view(1, nclasses, 1, 1)
    pred_bin = (prediction_argmax.view(nsamples, 1, height, width) == classes)
    gt_bin = (gt.view(nsamples, 1, height, width) == classes)
    intersection = (pred_bin * gt_bin).float().sum(dim=-2).sum(dim=-1)
    union = ((pred_bin + gt_bin) > 0).float().sum(dim=-2).sum(dim=-1)
    return (intersection + 1e-7) / (union + 1e-7)
