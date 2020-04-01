Project: 'Ray Tracing' a Vector Image
=====================================

> Marten de Vries

The ray tracing assignments made me wonder if you could use a similar approach
to generate vector instead of raster images. This requires describing all
possible rays instead of specific rays. This project explores that idea in the
form of an (incomplete) Python prototype.

I disqualify myself from the competition, because this submission uses tools
other than those allowed.


Demonstration
-------------

Two scenes as rendered by the prototype:

![a simple test scene](https://github.com/marten-de-vries/surfacetracer/blob/master/out-surface.png)

![a more complicated scene](https://github.com/marten-de-vries/surfacetracer/blob/master/out-surface-complex.png)

The first is a simple test scene, while the second demonstrates meshes and
succesful implementation of a BSP tree.


Limitations
-----------

- Triangles (& quads & meshes) only
- Incomplete/incorrect culling/clipping
- No perspective correction implemented (compare the image below, which was rendered using a ray tracer, with the one above. The shadows differ slightly). This should be relatively easy to fix.

![ray-traced version of the simple test scene](https://github.com/marten-de-vries/surfacetracer/blob/master/out.png)

None of these limitations are fundamental, I think.


Approach
--------

The final prototype (surfacetracer2.py) actually looks a lot like a rasterizing
pipeline: the vertices of triangles are projected to their location on the
'screen'. A difference is that this is done recursively for shadow rays: here
the triangle becomes the 'screen'. The same approach could be used to implement
reflection & refraction. Another difference is that a z-buffer cannot be used
to determine visibility (as that would require rasterization at a 'vector
image' stage), so a Binary Space Partitioning tree is used instead.

The vector image format is implicit in this demo (it's a bunch of classes that
could be serialized to disk), but varying the 'scaling_factor' parameter shows
that the demo is succesful: the same internal (vector) representation of what
is to be on the screen can be rendered at different resolutions.

An earlier attempt is in the surfacetracer.py & viewer.html) files. It failed
because it passed off too much to OpenGL, making the perspective correction
problem that the final version also shows a fundamental issue, as opposed to
one that just wasn't fixed.

Another even earlier attempt is in the 'surfacetracer (kopie).py' file. It tried
to directly write .svg vector files, but this is impractical when using the
Phong illumination model.

Finally, raytracer.py implements a reference raytracer in Python. It's strongly
inspired by the one we've worked with in this course.


Performance
-----------

It's about the same as the Python implementation of the Raytracer. The 'vector
image' format could be changed to allow for faster lookups, though. Currently
it's just an array of things on the screen ordered by visibility. This is the
current bottleneck.


Try it out
----------

```bash
python3 -m venv venv
. venv/bin/activate
pip install tqdm numpy matplotlib pywavefront
python3 surfacetracer2.py
```
