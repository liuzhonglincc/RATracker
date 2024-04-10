import lap
import numpy as np
from cython_bbox import bbox_overlaps as bbox_ious
from torch.utils.data import Dataset

class DatasetTrain(Dataset):
    def __init__(self, data_all):
        self.data_all = data_all

    def __getitem__(self, idx):
        return self.data_all[idx]

    def __len__(self):
        return len(self.data_all)
    
def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float),
        np.ascontiguousarray(btlbrs, dtype=np.float)
    )

    return ious
    
def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b

def Giou_np(bbox_p, bbox_g):
        """
        :param bbox_p: predict of bbox(N,4)(x1,y1,x2,y2)
        :param bbox_g: groundtruth of bbox(N,4)(x1,y1,x2,y2)
        :return:
        example:
        p = np.array([[21,45,103,172],
                    [34,283,155,406],
                    [202,174,271,255]])
        g = np.array([[59,106,154,230],
                    [71,272,191,419],
                    [257,244,329,351]])
        """
        # for details should go to https://arxiv.org/pdf/1902.09630.pdf
        # ensure predict's bbox form
        x1p = np.minimum(bbox_p[0], bbox_p[2])
        x2p = np.maximum(bbox_p[0], bbox_p[2])
        y1p = np.minimum(bbox_p[1], bbox_p[3])
        y2p = np.maximum(bbox_p[1], bbox_p[3])

        bbox_p = [x1p, y1p, x2p, y2p]
        # calc area of Bg
        area_p = (bbox_p[2] - bbox_p[0]) * (bbox_p[3] - bbox_p[1])
        # calc area of Bp
        area_g = (bbox_g[2] - bbox_g[0]) * (bbox_g[3] - bbox_g[1])

        # cal intersection
        x1I = np.maximum(bbox_p[0], bbox_g[0])
        y1I = np.maximum(bbox_p[1], bbox_g[1])
        x2I = np.minimum(bbox_p[2], bbox_g[2])
        y2I = np.minimum(bbox_p[3], bbox_g[3])
        I = np.maximum((y2I - y1I), 0) * np.maximum((x2I - x1I), 0)

        # find enclosing box
        x1C = np.minimum(bbox_p[0], bbox_g[0])
        y1C = np.minimum(bbox_p[1], bbox_g[1])
        x2C = np.maximum(bbox_p[2], bbox_g[2])
        y2C = np.maximum(bbox_p[3], bbox_g[3])

        # calc area of Bc
        area_c = (x2C - x1C) * (y2C - y1C)
        U = area_p + area_g - I
        iou = 1.0 * I / U

        # Giou
        # giou = iou - (area_c - U) / area_c

        giou = iou - (area_c - U) / area_c +  1.0 * I / area_p

        return giou