import matplotlib.pyplot as plt
import numpy
from tqdm.contrib.concurrent import process_map

import collections
import contextlib
import dataclasses
import functools
import itertools
import json
import os
import sys
import typing

from helpers import Point, Vector, Color, Texture, Material, normalized, length
from raytracer import Light, Ray
from surfaces import Mesh, BoundingBox


# TODO: remove
numpy.random.seed(1)

# TODO: maybe move into classes?
EPSILON = 0.001


@dataclasses.dataclass
class Triangle:
    vertices: typing.List[Point]
    material: Material
    n: Vector = dataclasses.field(init=False)

    def __post_init__(self):
        a, b, c = self.vertices
        self.n = normalized(numpy.cross(b - a, c - a))

    def planetest(self, p):
        # gives ~ distance from plane
        return numpy.dot(self.n, p - self.vertices[0])

    def project(self, viewInfo):
        vertices = [viewInfo.project(p) for p in self.vertices]
        return Triangle2D(vertices)

    def barycentric(self, p):
        # source: https://gamedev.stackexchange.com/a/23745

        # calculate beta & gamma

        v0 = self.vertices[1] - self.vertices[0]
        v1 = self.vertices[2] - self.vertices[0]
        v2 = p - self.vertices[0]
        d00 = numpy.dot(v0, v0)
        d01 = numpy.dot(v0, v1)
        d11 = numpy.dot(v1, v1)
        d20 = numpy.dot(v2, v0)
        d21 = numpy.dot(v2, v1)
        denom = d00 * d11 - d01 * d01
        gamma = (d00 * d21 - d01 * d20) / denom
        beta = (d11 * d20 - d01 * d21) / denom

        return Point2D(beta, gamma)


def Point2D(x_or_beta, y_or_gamma):
    return numpy.array([x_or_beta, y_or_gamma])


class Surfaces2D:
    # TODO: accelerate somehow
    def __init__(self):
        self.objects = []

    def occlusions(self, other_triangle):
        result = []
        for object in self.objects:
            isintersected_a, isintersected_b = itertools.tee(
                object.intersect_point(p)
                for p in other_triangle.vertices
            )
            if all(isintersected_a):
                return True, [object]
            if any(isintersected_b):
                result.append(object)
        return False, result

    def add(self, triangle):
        """Invariant: the closest objects should be added first."""

        self.objects.append(triangle)

    def __getitem__(self, coord):
        for object in self.objects:
            hit = object.intersect_point(coord)
            if hit:
                return hit


class Light2D(typing.NamedTuple):
    ls: typing.List[Vector]
    obstructions: Surfaces2D
    I: Color


@dataclasses.dataclass
class Triangle2D:
    vertices: typing.List[Point2D]
    ds: typing.List[Vector] = dataclasses.field(init=False, repr=False)
    n: Vector = dataclasses.field(init=False, repr=False)
    material: Material = dataclasses.field(init=False, repr=False)
    lights: typing.List[Light2D] = dataclasses.field(init=False, repr=False)

    def interpolate(self, ps, coords):
        a, b, c = ps
        beta, gamma = coords
        alpha = 1 - gamma - beta
        return a * alpha + b * beta + c * gamma

    def barycentric(self, p):
        # source: https://gamedev.stackexchange.com/a/63203

        # calculate beta & gamma based on 2d coordinates in p
        v0 = self.vertices[1] - self.vertices[0]
        v1 = self.vertices[2] - self.vertices[0]
        v2 = p - self.vertices[0]

        denom = v0[0] * v1[1] - v1[0] * v0[1]
        beta = (v2[0] * v1[1] - v1[0] * v2[1]) / denom
        gamma = (v0[0] * v2[1] - v2[0] * v0[1]) / denom

        # TODO: convert barycentric coordinates from screen to world space!
        # now the perspective isn't quite right.
        return Point2D(beta, gamma)

    def intersect_point(self, p):
        beta, gamma = self.barycentric(p)

        if not (0 <= gamma <= 1):
            return None
        if not (0 <= beta <= 1 - gamma):
            return None

        return Hit(self, Point2D(beta, gamma))


class Hit(typing.NamedTuple):
    surface: Triangle
    barycentric: Point2D

    def interpolate(self, p):
        return self.surface.interpolate(p, self.barycentric)


class BSPTree:
    def __init__(self, triangle):
        self.triangle = triangle
        self.minus = None
        self.plus = None

        min = numpy.min(self.triangle.vertices, axis=0)
        max = numpy.max(self.triangle.vertices, axis=0)
        self.bounding_box = BoundingBox(min, max)

    def f(self, p):
        return self.triangle.planetest(p)

    def dot(self):
        iter = itertools.count()
        names = collections.defaultdict(functools.partial(next, iter))
        yield 'digraph {'
        for line in self.dot_node(names):
            yield ' ' * 4 + line
        yield '}'

    @property
    def height(self):
        return max(
            getattr(self.minus, 'height', 0),
            getattr(self.plus, 'height', 0)
        ) + 1

    def dot_node(self, names):
        for child in [self.minus, self.plus]:
            if child:
                yield f'{names[self]} -> {names[child]};'
                yield from child.dot_node(names)

    @classmethod
    def from_list(cls, triangles):
        triangles = list(triangles)
        # TODO: make use of free splits (see Computational Geometry: Algorithms
        # and Applications, Chapter 12.4)
        numpy.random.shuffle(triangles)
        print('triangles in', len(triangles))
        tree = cls(triangles[0])
        for triangle in triangles[1:]:
            tree.add(triangle)
        print('height', tree.height)
        print('triangles out', len(tree))
        with open('tree.dot', 'w') as f:
            f.writelines(tree.dot())
        return tree

    def add(self, triangle):
        a, b, c = triangle.vertices
        fa = self.round_close_to_zero(self.f(a))
        fb = self.round_close_to_zero(self.f(b))
        fc = self.round_close_to_zero(self.f(c))
        if fa <= 0 and fb <= 0 and fc <= 0:
            self.minus = self.add_to(self.minus, triangle)
        elif fa >= 0 and fb >= 0 and fc >= 0:
            self.plus = self.add_to(self.plus, triangle)
        else:
            # split triangle - first force c to one side of the plane and a & b
            # to the other
            if fa * fc >= 0:
                fa, fb, fc = fc, fa, fb
                a, b, c = c, a, b
            elif fb * fc >= 0:
                fa, fb, fc = fb, fc, fa
                a, b, c = b, c, a

            # calculate the splitting points
            D = - numpy.dot(self.triangle.n, self.triangle.vertices[0])
            ta = - (numpy.dot(self.triangle.n, a) + D) / numpy.dot(self.triangle.n, c - a)
            A = a + ta * (c - a)
            tb = - (numpy.dot(self.triangle.n, b) + D) / numpy.dot(self.triangle.n, c - b)
            B = b + tb * (c - b)

            # create the new triangles - handling zero-area cases.
            if not numpy.allclose(a, A, EPSILON):
                self.add(Triangle([a, b, A], triangle.material))
            if not numpy.allclose(b, B, EPSILON):
                self.add(Triangle([b, B, A], triangle.material))
            self.add(Triangle([A, B, c], triangle.material))

    def round_close_to_zero(self, num):
        if abs(num) < EPSILON:
            return 0
        return num

    def add_to(self, branch, triangle):
        try:
            branch.add(triangle)
        except AttributeError:
            branch = BSPTree(triangle)
        self.bounding_box = BoundingBox.combine(self.bounding_box,
                                                branch.bounding_box)
        return branch

    def walk(self, e, back_to_front=True, ignore=lambda bbox: False):
        """Walk the BSP tree from one side to the other from the PoV e. Filter
        the tree using the ignore function.

        """
        if ignore(self.bounding_box):
            return
        direction = 1 if back_to_front else -1
        if direction * self.f(e) < 0:
            before, after = self.plus, self.minus
        else:
            before, after = self.minus, self.plus
        if before:
            yield from before.walk(e, back_to_front, ignore)
        yield self.triangle
        if after:
            yield from after.walk(e, back_to_front, ignore)

    def __len__(self):
        return (
            (len(self.minus) if self.minus is not None else 0) +
            (len(self.plus) if self.plus is not None else 0) +
            1
        )


def trace_silhouettes(surfaces, viewInfo):
    # projects every surface in surfaces, determines if it's visible, and
    # if so, stores it in surfaces_2d.
    originals = {}
    surfaces_2d = Surfaces2D()
    for surface in viewInfo.from_front_to_back(surfaces):
        # a flat 2d surface
        try:
            projected = surface.project(viewInfo)
        except ValueError:
            # TODO: FIXME: clipping/culling error. Ignore for now.
            continue
        if viewInfo.fully_outside_view(projected):
            continue
        # TODO: make sure that in the next recursion, the part of the triangle
        # outside the frustum is 'covered' by 'occlusion surfaces', i.e.
        # silhouette-only surfaces that surface_2d gets initialized with, such
        # that anything behind is not recursed into.
        # assert not viewInfo.partially_outside_view(projected):

        # TODO: also handle partial occlusion (see above) - including? the case
        # were multiple triangles occlude the triangle fully, but only
        # together.
        fully_occluded, occluders = surfaces_2d.occlusions(projected)
        if not fully_occluded:
            surfaces_2d.add(projected)
            originals[id(projected)] = surface
    return surfaces_2d, originals


def trace_surfaces(scene, viewInfo, depth):
    surfaces_2d, originals = trace_silhouettes(scene.surfaces, viewInfo)

    for surface in surfaces_2d.objects:
        orig = originals[id(surface)]
        surface.ds = [normalized(p - viewInfo.e) for p in orig.vertices]
        surface.n = orig.n
        surface.material = orig.material

        # by now, we're certain that these surfaces are in the final image, so
        # gather more info about them.
        surface.lights = []
        for l in scene.lights:
            ls = [normalized(l.position - p) for p in orig.vertices]
            if scene.render_shadows:
                obstructions, _ = trace_silhouettes(scene.surfaces, ShadowViewInfo(orig, l.position))
            else:
                obstructions = Surfaces2D()
            surface.lights.append(Light2D(ls, obstructions, l.I))
        # TODO: enable
        # if surface.material.ks > 0 and depth > 0:
        #     surface.reflection = trace_surfaces(scene, ViewInfo.reflect(viewInfo, surface), depth - 1)
        # TODO: similar for refraction
        # TODO: handle depth

    return surfaces_2d


class Scene(typing.NamedTuple):
    eye: Point
    surfaces: BSPTree
    lights: typing.Iterable[Light]
    render_shadows: bool

    @classmethod
    def load_from_json(cls, json_path):
        with open(json_path) as f:
            data = json.load(f)
        lights = [Light(Color(*l['color']), Point(*l['position']))
                  for l in data['Lights']]
        surfaces = []
        for desc in data['Objects']:
            descm = desc['material']
            col = Color(*descm.get('color', [1, 1, 1]))
            try:
                tex = Texture(path(descm['texture']))
            except KeyError:
                tex = None
            material = Material(ka=descm['ka'] * col, kd=descm['kd'] * col,
                                ks=descm['ks'] * Color(1, 1, 1), p=descm['n'],
                                texture=tex)

            if desc['type'] == 'quad':
                surfaces.append(Triangle([
                    Point(*desc['v0']),
                    Point(*desc['v1']),
                    Point(*desc['v2']),
                ], material))
                surfaces.append(Triangle([
                    Point(*desc['v0']),
                    Point(*desc['v2']),
                    Point(*desc['v3']),
                ], material))
            elif desc['type'] == 'mesh':
                m = Mesh(path(desc['filename']),
                         position=Point(*desc['position']),
                         rotation=Vector(*desc['rotation']),
                         scale=Vector(*desc['scale']), material=material)
                for surface in m.surfaces:
                    surfaces.append(Triangle([
                        surface.a,
                        surface.b,
                        surface.c,
                    ], material))
            elif desc['type'] == 'triangle':
                surfaces.append(Triangle([
                    Point(*desc['v0']),
                    Point(*desc['v1']),
                    Point(*desc['v2']),
                ], material))
            else:
                raise ValueError(f"Unimplemented type: {desc['type']}")
        tree = BSPTree.from_list(surfaces)
        render_shadows = data.get('Shadows', False)
        return cls(Point(*data['Eye']), tree, lights, render_shadows)


class ViewInfo(typing.NamedTuple):
    e: Point  # the eye
    width: int
    height: int
    d: float

    def fully_outside_view(self, surface):
        # assumes projection happened
        return (
            all(p[0] < 0 for p in surface.vertices) or
            all(p[0] > self.width for p in surface.vertices) or
            all(p[1] < 0 for p in surface.vertices) or
            all(p[1] > self.height for p in surface.vertices)
        )

    def partially_outside_view(self, surface):
        # assumes projection happened
        return not all(
            0 <= p[0] <= self.width and
            0 <= p[1] <= self.height
            for p in surface.vertices
        )

    def from_front_to_back(self, surfaces):
        return surfaces.walk(self.e, back_to_front=False,
                             ignore=self.outside_frustum)

    def outside_frustum(self, bbox):
        topleft = Point(0, self.height, self.d)
        topright = Point(self.width, self.height, self.d)
        bottomleft = Point(0, 0, self.d)
        bottomright = Point(self.width, 0, self.d)
        # convention: all normals point inwards
        planes = [
            # top
            Triangle([self.e, topleft, topright], None),
            # bottom
            Triangle([self.e, bottomright, bottomleft], None),
            # left
            Triangle([self.e, bottomleft, topleft], None),
            # right
            Triangle([self.e, topright, bottomright], None),
        ]
        x = test_outside(bbox, planes)
        return x

    def project(self, point):
        point_on_screen = Point(0, 0, self.d)
        normal = Point(0, 0, 1)
        numerator = numpy.dot(point_on_screen - self.e, normal)

        ray = Ray(self.e, normalized(point - self.e))
        t = numerator / numpy.dot(ray.d, normal)
        if t < 0:
            raise ValueError("Not enough clipping/culling")
        x, y, z = ray.at(t)
        return Point2D(x, self.height - y)


def test_outside(bbox, planes):
    for plane in planes:
        for point in itertools.product([bbox.min, bbox.max], repeat=3):
            corner_point = Point(point[0][0], point[1][1], point[2][2])
            if plane.planetest(corner_point) > 0:
                # at least (partly) inside this plane
                break
        else:
            # all corners are fully outside this plane
            return True
    return False


@dataclasses.dataclass
class ShadowViewInfo:
    orig: Triangle
    l: Light

    def __post_init__(self):
        self.e = self.l
        self.height = 400
        self.width = 400

    def project(self, point):
        if self.orig.planetest(point) < EPSILON:
            raise ValueError("Not enough clipping/culling")
        # TODO: refactor commonality with other ViewInfo class
        numerator = numpy.dot(self.orig.vertices[0] - self.l, self.orig.n)

        ray = Ray(self.l, normalized(point - self.l))
        t = numerator / numpy.dot(ray.d, self.orig.n)
        if t < 0:
            raise ValueError("Not enough clipping/culling")

        return self.orig.barycentric(ray.at(t))

    def fully_outside_view(self, surface):
        return all(
            beta < 0
            for beta, _ in surface.vertices
        ) or all(
            gamma < 0
            for _, gamma in surface.vertices
        ) or all(
            1 - gamma - beta < 0  # alpha < 0
            for beta, gamma in surface.vertices
        )

    def partially_outside_view(self, surface):
        return not all(
            0 <= gamma <= 1 and
            0 <= beta <= 1 - gamma
            for beta, gamma in surface.vertices
        )

    def from_front_to_back(self, surfaces):
        return surfaces.walk(self.l, back_to_front=True,
                             ignore=self.outside_frustum)

    def outside_frustum(self, bbox):
        a, b, c = self.orig.vertices
        # convention: all normals point inwards
        planes = [
            # plane from light to triangle corners
            Triangle([self.l, b, a], None),
            Triangle([self.l, c, b], None),
            Triangle([self.l, a, c], None),
            # the triangle itself # TODO: check if normal is in the right direction!
            Triangle([v + self.orig.n * EPSILON for v in self.orig.vertices], None),
        ]
        return test_outside(bbox, planes)


def path(filename):
    basepath = "/home/marten/AI5/ComputerGraphics/assignments/raytracer-lab1/build"
    return os.path.join(basepath, filename)


@dataclasses.dataclass
class Rasterizer:
    surfaces_2d: Surfaces2D
    viewInfo: ViewInfo
    scale_factor: float = 1

    def __post_init__(self):
        self.out_height = int(self.viewInfo.height * self.scale_factor)
        self.out_width = int(self.viewInfo.width * self.scale_factor)

    def rasterize(self, filename, multiprocess):
        # This could perhaps be implemented as a fragment shader (would require quite
        # big uniforms, but the rendering process could be split out over multiple
        # calls I guess...)

        if multiprocess:
            out = process_map(self.calculate_row, range(self.out_height))
        else:
            out = list(map(self.calculate_row, range(self.out_height)))
        out = numpy.reshape(out, (self.out_height, self.out_width, 3))
        plt.imsave(filename, out)

    def calculate_row(self, y):
        row = numpy.zeros((self.out_width, 3))
        for x in range(self.out_width):
            virtual_x = (x + 0.5) / self.scale_factor
            virtual_y = (y + 0.5) / self.scale_factor

            hit = self.surfaces_2d[virtual_x, virtual_y]
            row[x, :] = self.color(hit)
        return row

    def color(self, hit):
        if not hit:
            return Color(0, 0, 0)
        material = hit.surface.material
        n = hit.surface.n
        d = hit.interpolate(hit.surface.ds)
        if numpy.dot(n, d) > 0:
            n = -n

        # ambient
        L = material.ka.copy()
        for light in hit.surface.lights:
            if light.obstructions[hit.barycentric]:
                continue
            l = hit.interpolate(light.ls)

            r = 2 * numpy.dot(n, l) * n - l
            # diffuse
            L += material.kd * light.I * max(0, numpy.dot(n, l))
            # specular
            L += material.ks * light.I * max(0, numpy.dot(r, -d))**material.p

        return numpy.clip(L, 0, 1)


def main():
    try:
        json_path = sys.argv[1]
    except IndexError:
        json_path = 'in.json'
    scene = Scene.load_from_json(json_path)
    camera = ViewInfo(scene.eye, 400, 400, 0)
    # camera = ShadowViewInfo(Triangle([
    #     Point(0, 0, 0),
    #     Point(400, 0, 0),
    #     Point(0, 400, 0)
    # ], None), Point(200, 200, 1000))
    surfaces_2d = trace_surfaces(scene, camera, depth=0)

    # surfaces_2d now contains our vector image format. Given an x & a y, it
    # should return the closest surface at that point (as determined by the
    # insertion order). We could store it to disk easily.

    rasterizer = Rasterizer(surfaces_2d, camera)
    rasterizer.rasterize('out-surface.png', multiprocess=True)


if __name__ == '__main__':
    main()
