import matplotlib.pyplot as plt
import numpy

import typing


# not really types, but it's too much effort to properly subclass just for mypy
def Point(x, y, z):
    return numpy.array([x, y, z])


def Vector(x, y, z):
    return numpy.array([x, y, z])


def Color(r, g, b):
    return numpy.array([r, g, b])


class Texture:
    def __init__(self, path):
        self.data = plt.imread(path)

    def __getitem__(self, coord):
        u, v = coord
        h, w, _ = self.data.shape
        x = int(u * (w - 1))
        y = int(v * (h - 1))
        return Color(*self.data[y, x])


class Material(typing.NamedTuple):
    ka: Color
    kd: Color
    ks: Color
    p: int
    texture: typing.Optional[Texture]


def normalized(x):
    return x / length(x)


def length(x):
    return numpy.sqrt(numpy.dot(x, x))
