
def s_membership(box, sample, sensitivity):
    if box.contains(sample):
        return 1
    d = box.dim
    val = 0
    for i in range(d):
        if sample[i] == 0:
            continue
        s_range = sensitivity * abs(box.W[i] - box.V[i])
        a_v = box.V[i] - s_range
        b_v = box.V[i]
        val_v = 0
        if a_v != b_v:
            m_v = box.V[i] - s_range / 2.0
            if m_v >= sample[i] >= a_v:
                val_v = 2 * ((sample[i] - a_v) / (b_v - a_v))**2
            elif b_v >= sample[i] > m_v:
                val_v = 1 - 2 * ((sample[i] - b_v) / (b_v - a_v)) ** 2
        a_w = box.W[i]
        b_w = box.W[i] + s_range
        val_w = 0
        if a_w != b_w:
            m_w = box.W[i] + s_range / 2.0
            if m_w >= sample[i] >= a_w:
                val_w = 2 * ((sample[i] - a_w) / (b_w - a_w)) ** 2
            elif b_w >= sample[i] > m_w:
                val_w = 1 - 2 * ((sample[i] - b_w) / (b_w - a_w)) ** 2
        val = min(val, max(val_v, val_w))
        if val == 0:
            break
    assert 1.0 >= val >= 0
    return val


def trap_membership(box, sample, sensitivity):
    if box.contains(sample):
        return 1
    d = box.dim
    val = 0
    for i in range(d):
        if sample[i] == 0:
            continue
        trap_range = sensitivity * abs(box.W[i] - box.V[i])
        a = box.V[i] - trap_range
        b = box.W[i] + trap_range
        if a == b:
            continue
        val_tmp = 0
        if box.V[i] >= sample[i] >= a:
            val_tmp = (sample[i] - a) / (box.V[i] - a)
        elif b >= sample[i] > box.W[i]:
            val_tmp = (b - sample[i]) / (b - box.W[i])
        val += val_tmp
    val /= d
    assert 1.0 >= val >= 0
    return val


def simpsons_membership(box, sample, sensitivity):
    d = box.dim
    b = 1.0
    for i in range(d):
        if sample[i] == 0:
            continue
        b = min(b, 0.5 * (max(0, 1 - max(0, sensitivity * min(1, sample[i] - box.W[i])))
                    + max(0, 1 - max(0, sensitivity * min(1, box.V[i] - sample[i]))))
                )
        if b == 0:
            return b
    assert 1.0 >= b >= 0
    return b


def min_max_membership(box, sample, sensitivity):
    def f(x, gamma):
        if x * gamma > 1:
            return 1
        elif x * gamma > 0:
            return x * gamma
        else:
            return 0
    d = box.dim
    b = 1
    for i in range(d):
        if sample[i] == 0:
            continue
        b = min(b, min(1.0-f(sample[i] - box.W[i], sensitivity),
                           1.0-f(box.V[i] - sample[i], sensitivity))
                )
        if b == 0:
            break
        assert 1.0 >= b >= 0
    assert 1.0 >= b >= 0
    return b


def get_overlap_data(data, ols, label_val, class_label_idx, sample_val_idx):
    rv = [sample for sample in data if sample[class_label_idx] == label_val and ols.contains(sample[sample_val_idx])]
    return rv


def get_overlapping_coord(box1, box2):
    overlaps_V = []
    overlaps_W = []
    d = min(box1.dim, box2.dim)
    for i in range(d):
        overlaps_V.append(max(box1.V[i], box2.V[i]))
        overlaps_W.append(min(box1.W[i], box2.W[i]))
    return overlaps_V, overlaps_W

class HyperBox:

    def __init__(self, n, max_size, val=None):
        self.dim = n
        self.max_size = max_size
        if val is None:
            self.V = [0 for _ in range(self.dim)]
            self.W = [0 for _ in range(self.dim)]
        else:
            self.dim = len(val)
            self.V = [val[i] for i in range(self.dim)]
            self.W = [val[i] for i in range(self.dim)]

    def contains(self, vals):
        d  = min(len(vals), self.dim)
        for i in range(d):
            if vals[i] == 0:
                continue
            if vals[i] < self.V[i] or vals[i] > self.W[i]:
                return False
        return True

    def overlaps(self, box):
        overlap = True
        d = min(box.dim, self.dim)
        for i in range(d):
            if not (box.W[i] >= self.V[i] and self.W[i] >= box.V[i]):
                overlap = False
                break
        return overlap

    def can_expand(self, vals):
        d = min(len(vals), self.dim)
        for i in range(d):
            if vals[i] == 0:
                continue
            if max(vals[i], self.W[i]) - min(vals[i], self.V[i]) > self.max_size:
                return False
        return True

    def expand(self, vals):
        d = min(len(vals), self.dim)
        for i in range(d):
            if vals[i] == 0:
                continue
            self.V[i] = min(self.V[i], vals[i])
            self.W[i] = max(self.W[i], vals[i])

    def print_info(self):
        print('HyperBox of dim ' + str(self.dim))
        print(self.V)
        print(self.W)