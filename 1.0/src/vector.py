import math


class Vector:
    attr = []

    def __init__(self, attr, label):
        self.attr = attr
        self.label = label

    def __str__(self):
        s = '['
        for i in range(len(self.attr) - 1):
            s += str(str(self.attr[i])) + ', '
        assert len(self.attr) > 0
        s += str(self.attr[len(self.attr) - 1]) + ', ' + self.label + ']'
        return s

    def __sub__(self, other):
        assert len(self.attr) == len(other.attr)
        attr = [self.attr[i] - other.attr[i] for i in range(len(self.attr))]
        return Vector(attr, '')

    def __add__(self, other):
        assert len(self.attr) == len(other.attr)
        attr = [self.attr[i] + other.attr[i] for i in range(len(self.attr))]
        return Vector(attr, '')

    def __ne__(self, other):
        dist = self.distTo(other)
        return dist > 0.00001

    def distTo(self, other):
        x = self - other
        return math.sqrt(sum(i ** 2 for i in x.attr))
