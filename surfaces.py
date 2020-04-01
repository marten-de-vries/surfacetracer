import numpy
import pywavefront

import abc
import dataclasses
import functools
import itertools
import typing

from helpers import Point, Vector, Material, normalized


class Surface(abc.ABC):
    pass


class Hit(typing.NamedTuple):
    t: float
    n: Vector


# TODO: fix implementation, because this one doesn't work...
class BSPNode(Surface):
    def __init__(self, surfaces, depth=0):
        self.axis = depth % 3

        surfs = sorted(surfaces, key=lambda s: s.bounding_box.min[self.axis])
        self.D = surfs[len(surfs) // 2].bounding_box.min[self.axis]

        surfs_left = [s for s in surfs if s.bounding_box.min[self.axis] <= self.D]
        surfs_right = [s for s in surfs if s.bounding_box.max[self.axis] > self.D]

        matches = sum(1 for l, r in itertools.product(surfs_left, surfs_right)
                      if l is r)
        if matches > len(surfs_left) // 2:
            self.left = Group(surfs_left)
        else:
            self.left = BSPNode(surfs_left, depth + 1)
        if matches > len(surfs_right) // 2:
            self.right = Group(surfs_right)
        else:
            self.right = BSPNode(surfs_right, depth + 1)
        self.bounding_box = BoundingBox.combine(self.left.bounding_box,
                                                self.right.bounding_box)

    def __repr__(self):
        return "<%s %r %r %r>" % (self.__class__.__name__, self.D, self.left,
                                  self.right)

    def intersect(self, ray, tmin, tmax):
        u_p = ray.e[self.axis] + tmin * ray.d[self.axis]
        if u_p < self.D:
            args = lambda u_b: u_b < 0, self.left, self.right
        else:
            args = lambda u_b: u_b > 0, self.right, self.left
        return self.handle_case(ray, tmin, tmax, *args)

    def handle_case(self, ray, tmin, tmax, test, left, right):
        # var names like we're processing the u_p < D case, but things can
        # actually be reversed.
        left_hit = self.left.intersect(ray, tmin, tmax)
        if test(ray.d[self.axis]):
            return left_hit
        t = (self.D - ray.e[self.axis]) / ray.d[self.axis]
        if t > tmax or left_hit[0]:
            return left_hit
        return self.right.intersect(ray, tmin, tmax)


class Group(Surface):
    def __init__(self, surfaces):
        bboxes = (s.bounding_box for s in surfaces)
        self.bounding_box = functools.reduce(BoundingBox.combine, bboxes)
        self.surfaces = surfaces

    def intersect(self, ray, tmin, tmax):
        best, bestsurface = None, None
        for surface in self.surfaces:
            # possibly surface != hit_surface: when a surface groups smaller
            # surfaces
            hit, hit_surface = surface.intersect(ray, tmin, tmax)
            if hit:
                tmax = hit.t
                best, bestsurface = hit, hit_surface
        return best, bestsurface

    def __repr__(self):
        return "<%s %r>" % (self.__class__.__name__, self.surfaces)


class BoundingBox(typing.NamedTuple):
    min: Point
    max: Point

    @classmethod
    def combine(cls, a, b):
        mins, maxs = zip(a, b)
        return cls(numpy.min(mins, axis=0), numpy.max(maxs, axis=0))


class Quad(Group):
    def __init__(self, a, b, c, d, material):
        super().__init__([Triangle(a, b, c, material),
                          Triangle(a, c, d, material)])


class Mesh(Group):
    def __init__(self, filename, position, rotation, scale, material):
        transformation = numpy.array([
            # translation
            [1, 0, 0, position[0]],
            [0, 1, 0, position[1]],
            [0, 0, 1, position[2]],
            [0, 0, 0, 1],
        ]) @ numpy.array([
            # non-uniform scaling
            [scale[0], 0, 0, 0],
            [0, scale[1], 0, 0],
            [0, 0, scale[2], 0],
            [0, 0, 0, 1],
        ]) @ numpy.array([
            # rotation around z
            [numpy.cos(rotation[2]), -numpy.sin(rotation[2]), 0, 0],
            [numpy.sin(rotation[2]), numpy.cos(rotation[2]), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]) @ numpy.array([
            # rotation around y
            [numpy.cos(rotation[1]), 0, numpy.sin(rotation[1]), 0],
            [0, 1, 0, 0],
            [-numpy.sin(rotation[1]), 0, numpy.cos(rotation[1]), 0],
            [0, 0, 0, 1],
        ]) @ numpy.array([
            # rotation around x
            [1, 0, 0, 0],
            [0, numpy.cos(rotation[0]), -numpy.sin(rotation[0]), 0],
            [0, numpy.sin(rotation[0]), numpy.cos(rotation[0]), 0],
            [0, 0, 0, 1],
        ])

        scene = pywavefront.Wavefront(filename, collect_faces=True)
        surfaces = []
        for (ai, bi, ci) in scene.mesh_list[0].faces:
            a1 = scene.vertices[ai]
            b1 = scene.vertices[bi]
            c1 = scene.vertices[ci]

            a = Point(*(transformation @ numpy.array(a1 + (1,)))[:3])
            b = Point(*(transformation @ numpy.array(b1 + (1,)))[:3])
            c = Point(*(transformation @ numpy.array(c1 + (1,)))[:3])
            surfaces.append(Triangle(a, b, c, material))
        super().__init__(surfaces)


@dataclasses.dataclass
class Triangle(Surface):
    a: Point
    b: Point
    c: Point
    material: Material = dataclasses.field(repr=False)
    n: Vector = dataclasses.field(init=False, repr=False)
    bounding_box: BoundingBox = dataclasses.field(init=False, repr=False)

    def __post_init__(self):
        self.n = normalized(numpy.cross(self.b - self.a, self.c - self.a))

        min = numpy.min([self.a, self.b, self.c], axis=0)
        max = numpy.max([self.a, self.b, self.c], axis=0)
        self.bounding_box = BoundingBox(min, max)

    def intersect(self, ray, tmin, tmax):
        # from the text book: 'Ray triangle intersection' section.

        # xa - xb
        a = self.a[0] - self.b[0]
        # ya - yb
        b = self.a[1] - self.b[1]
        # za - zb
        c = self.a[2] - self.b[2]

        # xa - xc
        d = self.a[0] - self.c[0]
        # ya - yc
        e = self.a[1] - self.c[1]
        # za - zc
        f = self.a[2] - self.c[2]

        # xd
        # minus to account for a reversed order when calculating cross products
        # (probably, this works anyway)
        g = -ray.d[0]
        # xy
        h = -ray.d[1]
        # xz
        i = -ray.d[2]

        # xa - xe
        j = self.a[0] - ray.e[0]
        # ya - ye
        k = self.a[1] - ray.e[1]
        # za - ze
        l = self.a[2] - ray.e[2]

        ei_minus_hf = (e * i) - (h * f)
        gf_minus_di = (g * f) - (d * i)
        dh_minus_eg = (d * h) - (e * g)

        ak_minus_jb = (a * k) - (j * b)
        jc_minus_al = (j * c) - (a * l)
        bl_minus_kc = (b * l) - (k * c)

        M = (a * ei_minus_hf) + (b * gf_minus_di) + (c * dh_minus_eg)

        if M == 0:
            return None, None
        t = ((f * ak_minus_jb) + (e * jc_minus_al) + (d * bl_minus_kc)) / M
        if not (tmin < t < tmax):
            return None, None
        gamma = ((i * ak_minus_jb) + (h * jc_minus_al) + (g * bl_minus_kc)) / M
        if gamma < 0 or gamma > 1:
            return None, None
        beta = ((j * ei_minus_hf) + (k * gf_minus_di) + (l * dh_minus_eg)) / M
        if beta < 0 or beta > 1 - gamma:
            return None, None

        return Hit(t, self.n), self


@dataclasses.dataclass
class Sphere(Surface):
    c: Point
    R: float
    material: Material
    axis: Vector
    angle: float
    bounding_box: BoundingBox = dataclasses.field(init=False)

    def __post_init__(self):
        self.bounding_box = BoundingBox(self.c - self.R, self.c + self.R)

    def to_uv(self, hit_pos):
        k = normalized(self.axis)
        theta = -self.angle / 180 * numpy.pi
        p = self.rotated(hit_pos - self.c, k, theta)

        u = 0.5 + numpy.arctan2(p[1], p[0]) / (2 * numpy.pi)
        v = numpy.arccos(p[2] / self.R) / numpy.pi
        return u, v

    @staticmethod
    def rotated(v, k, theta):
        # formula source: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
        return (v * numpy.cos(theta) + numpy.cross(k, v) * numpy.sin(theta) +
                k * numpy.dot(k, v) * (1 - numpy.cos(theta)))

    def intersect(self, ray, tmin, tmax):
        origin = ray.e - self.c

        a = numpy.dot(ray.d, ray.d)
        b = 2 * numpy.dot(ray.d, origin)
        c = numpy.dot(origin, origin) - self.R**2

        for t in self.solve_quadratic_equation(a, b, c):
            if t < tmin:
                continue  # get to the promising candidates
            if t > tmax:
                break  # further candidates will not be good either
            p = ray.at(t)
            n = (p - self.c) / self.R

            return Hit(t, n), self
        return None, None

    @staticmethod
    def solve_quadratic_equation(a, b, c):
        # TODO: numeric stability? Or (if fast enough) just use decimals?
        discriminant = b**2 - 4 * a * c

        if discriminant == 0:
            return -b / (2 * a),
        elif discriminant < 0:
            return ()
        else:
            x1 = (-b + numpy.sqrt(discriminant)) / (2 * a)
            x2 = (-b - numpy.sqrt(discriminant)) / (2 * a)
            if x1 > x2:
                x1, x2 = x2, x1
            return x1, x2


# OpenCylinder, Disc, ClosedCylinder...
