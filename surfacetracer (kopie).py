# NOTE: Using svg is too ambitious, which is why this attempt was abandoned.

import numpy

import abc
import dataclasses

from helpers import Color, Point, Vector, normalized, Material, length
from raytracer import Ray

class Surface(abc.ABC):
    pass


def join_by_comma(point):
    return ', '.join(str(number) for number in point)


def to_svg_color(color):
    return f'rgb({ join_by_comma(color * 255) })'


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
        points = numpy.array([self.a, self.b, self.c])
        screen_points = [join_by_comma(point_to_screen(p)[:2]) for p in points]

# [[-250  520 1250]
#  [-550  520 1100]
#  [-400  280 1200]]
        ls = []
        print(light - (self.a + self.b + self.c) / 3)
        for point in points:
            l = light - point
            ls.append(l)
        print(numpy.mean(ls, axis=0))

        # so we can calculate for each pixel:
        # - n
        # - l
        # -

        midpoint = (self.a + self.b + self.c) / 3
        midpoint_screen = point_to_screen(midpoint)
        change = midpoint_screen - midpoint
        light_rel_screen = light + change
        x, y, z = light_rel_screen

        # TODO: make lighting color configurable!!!
        # TODO: same for specularConstant
        return f"""
            <defs>
                <filter id="phong">
                    <feSpecularLighting result="specular" lighting-color="white" specularConstant="1" specularExponent="5">
                        <fePointLight x="{x}" y="{y}" z="{z}" />
                    </feSpecularLighting>
                    <feComposite in="specular" in2="SourceGraphic" operator='arithmetic' k1='1' k2='0' k3='1' k4='0' />
                </filter>
            </defs>
            <polygon filter="url(#phong)" points="{' '.join(screen_points)}" fill='{to_svg_color(self.material.ka)}' />
        """.strip()


light = Point(-200, 600, 1500)
eye = Point(200, 200, 1000)
point_on_screen = Point(0, 0, 700)
normal = Point(0, 0, 1)
numerator = numpy.dot(point_on_screen - eye, normal)

def point_to_screen(point):
    ray = Ray(eye, normalized(point - eye))
    t = numerator / numpy.dot(ray.d, normal)
    x, y, z = ray.at(t)
    return x, 400 - y, z

with open('out.svg', 'w') as f:
    f.write('<svg width="400" height="400" xmlns="http://www.w3.org/2000/svg">')

    f.write('<rect width="100%" height="100%" fill="black" />')
    f.write(Triangle(
        a=Point(50, 80, 250),
        b=Point(350, 80, 400),
        c=Point(200, 320, 300),
        material=Material(
            ka=Color(0.3, 0, 0),
            kd=Color(0.7, 0, 0),
            ks=Color(0.5, 0, 0),
            p=5,
            texture=None
        ),
    ).vectorize())
    f.write('</svg>')
