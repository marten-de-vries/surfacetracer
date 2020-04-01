import matplotlib.pyplot as plt
import numpy
from tqdm.contrib.concurrent import process_map

import dataclasses
import json
import os
import sys
import typing

import surfaces
from helpers import Point, Vector, Color, Material, normalized, length, Texture


EPSILON = 1E-3


class Ray(typing.NamedTuple):
    e: Point
    d: Vector

    def at(self, t):
        return self.e + t * self.d


class Light(typing.NamedTuple):
    I: Color
    position: Point


class Scene(typing.NamedTuple):
    eye: Point
    surface: surfaces.Surface
    lights: typing.List[Light]
    render_shadows: bool

    @classmethod
    def load_from_json(cls, json_path):
        with open(json_path) as f:
            data = json.load(f)
        lights = [Light(Color(*l['color']), Point(*l['position']))
                  for l in data['Lights']]
        surfs = []
        for desc in data['Objects']:
            descm = desc['material']
            col = Color(*descm.get('color', [1, 1, 1]))
            try:
                tex = Texture(path(descm['texture']))
            except KeyError:
                tex = None
            material = Material(ka=descm['ka'] * col, kd=descm['kd'] * col,
                                ks=descm['ks'] * Color(1, 1, 1), p=descm['n'], texture=tex)
            to_surface = getattr(cls, f"to_{desc['type']}")
            surfs.append(to_surface(desc, material))
        group = surfaces.Group(surfs)
        render_shadows = data.get('Shadows', False)
        return cls(Point(*data['Eye']), group, lights, render_shadows)

    @staticmethod
    def to_sphere(desc, material):
        return surfaces.Sphere(c=Point(*desc['position']), R=desc['radius'],
                               axis=Vector(*desc.get('axis', [1, 0, 0])),
                               angle=desc.get('angle', 0), material=material)

    @staticmethod
    def to_triangle(desc, material):
        return surfaces.Triangle(a=Point(*desc['v0']), b=Point(*desc['v1']),
                                 c=Point(*desc['v2']), material=material)

    @staticmethod
    def to_quad(desc, material):
        return surfaces.Quad(a=Point(*desc['v0']), b=Point(*desc['v1']),
                             c=Point(*desc['v2']), d=Point(*desc['v3']),
                             material=material)

    @staticmethod
    def to_mesh(desc, material):
        return surfaces.Mesh(path(desc['filename']),
                             position=Point(*desc['position']),
                             rotation=Vector(*desc['rotation']),
                             scale=Vector(*desc['scale']), material=material)


def path(filename):
    basepath = "/home/marten/AI5/ComputerGraphics/assignments/raytracer-lab1/build"
    return os.path.join(basepath, filename)


@dataclasses.dataclass
class Raytracer:
    scene: Scene
    nx: int
    ny: int
    d: float

    def run(self, filename, multiprocess):
        if multiprocess:
            out = process_map(self.calculate_row, range(self.ny))
        else:
            out = list(map(self.calculate_row, range(self.ny)))
        out = numpy.reshape(out, (self.ny, self.nx, 3))
        plt.imsave(filename, out)

    def calculate_row(self, y):
        row = numpy.zeros((self.nx, 3))
        for x in range(self.nx):
            pixel = Point(x + 0.5, self.ny - 1 - y + 0.5, self.d)
            ray = Ray(e=self.scene.eye, d=normalized(pixel - self.scene.eye))
            hit, obj = self.scene.surface.intersect(ray, 0, float('inf'))
            row[x, :] = self.color(ray, hit, obj)
        return row

    def color(self, ray, hit, obj):
        if not hit:
            return Color(0, 0, 0)

        n = hit.n
        if numpy.dot(n, ray.d) > 0:
            n = -n
        surf_pos = ray.at(hit.t)
        mat = obj.material

        textureColor = mat.texture[obj.to_uv(surf_pos)] if mat.texture else 1

        # ambient
        L = mat.ka * textureColor
        for light in self.scene.lights:
            towards_light = light.position - surf_pos
            l = normalized(towards_light)

            if self.scene.render_shadows:
                shadow_ray = Ray(surf_pos, l)
                tmax = length(towards_light)
                if self.scene.surface.intersect(shadow_ray, EPSILON, tmax)[0]:
                    continue
            # diffuse
            L += mat.kd * light.I * max(0, numpy.dot(n, l)) * textureColor

            # specular
            # Blinn-Phong (follows book):
            # h = normalized(l - ray.d)
            # L += mat.ks * light.I * max(0, numpy.dot(n, h))**mat.p

            # Phong (follows assignment):
            r = 2.0 * n.dot(l) * n - l
            L += mat.ks * light.I * max(0, numpy.dot(r, -ray.d))**mat.p
        return numpy.clip(L, 0, 1)


def main():
    try:
        json_path = sys.argv[1]
    except IndexError:
        json_path = 'in.json'

    rt = Raytracer(Scene.load_from_json(json_path), nx=400, ny=400, d=0)
    rt.run('out.png', multiprocess=True)


if __name__ == '__main__':
    main()


# reflecting & refracting?
# quad, triangle w/ texture?
