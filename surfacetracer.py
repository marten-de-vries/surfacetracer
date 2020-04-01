# TODO: currently, we assume each triangle vertex is at the same distance, i.e.
# we don't take the fact that it's been projected into account, which will
# result in wrong interpolation of values over its face. It's a pretty
# fundamental issue, which renders the current 'use vertex shaders to
# rasterize' approach moot. surfacetracer2.py's approach does not have this
# fundamental limitation.

import numpy

import abc
import dataclasses

from helpers import Color, Point, Vector, normalized, Material, length
from raytracer import Ray

import json

class Surface(abc.ABC):
    pass


@dataclasses.dataclass
class Triangle(Surface):
    a: Point
    b: Point
    c: Point
    material: Material
    n: Vector = dataclasses.field(init=False)

    def __post_init__(self):
        self.n = normalized(numpy.cross(self.b - self.a, self.c - self.a))

    def vectorize(self):
        points = [self.a, self.b, self.c]
        screen_points = [point_to_screen(p)[:2] for p in points]

        towards_light = []
        towards_eye = []
        for point in points:
            towards_light.append(normalized(light - point))
            towards_eye.append(normalized(point - eye))

        # TODO: support more than a single surface
        return {
            'type': 'triangle',  # TODO: assertion on this
            'points': screen_points,
            'normal': self.n,
            'towards_light': towards_light,
            'towards_eye': towards_eye,
            'width': 400,
            'height': 400,
            'ka': self.material.ka,
            'kd': self.material.kd,
            'ks': self.material.ks,
            'p': self.material.p,
        }

light = Point(-200, 600, 1500)
eye = Point(200, 200, 1000)
point_on_screen = Point(0, 0, 0)
normal = Point(0, 0, 1)
numerator = numpy.dot(point_on_screen - eye, normal)

def point_to_screen(point):
    ray = Ray(eye, normalized(point - eye))
    t = numerator / numpy.dot(ray.d, normal)
    x, y, z = ray.at(t)
    return x, 400 - y, z

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

with open('out.js', 'w') as f:
    f.write('DATA = ');
    json.dump(Triangle(
        a=Point(50, 80, 250),
        b=Point(350, 80, 400),
        c=Point(200, 320, 300),
        material=Material(
            ka=Color(0.3, 0, 0),
            kd=Color(0.7, 0, 0),
            ks=Color(0.5, 0.5, 0.5),
            p=5,
            texture=None
        ),
    ).vectorize(), f, cls=NumpyEncoder)
    f.write(';\n')
