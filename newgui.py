import math
# from taichi_glsl import *

from typing import List, Tuple
import taichi as ti

# import taichi_glsl as ts

arch = ti.vulkan if ti._lib.core.with_vulkan() else ti.cuda

ti.init(arch=ti.cpu)
# ti.init(arch=arch)
res = 640
strong = res * 0.001

black = ti.math.vec3(0, 0, 0)
white = ti.math.vec3(1, 1, 1)

pixels = ti.Vector.field(3, dtype=ti.f32, shape=(res, res))

vertices = ti.Vector.field(2, dtype=ti.f32, shape=10)

for n in range(vertices.shape[0]):
    vertices[n] = (40 * n, 50 * n)


@ti.func
def hsv2rgb(c):
    k = ti.math.vec4([1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0])
    p = ti.math.fract(c.xxx + k.xyz) * 6.0 - k.www
    absp = ti.math.vec3([abs(p.x), abs(p.y), abs(p.z)])
    return c.z * ti.math.mix(k.xxx, ti.math.clamp(absp - k.xxx, 0.0, 1.0), c.y)


@ti.kernel
def paint_line_point(pos: ti.math.vec2, radius: ti.f32, strength: ti.f32):
    for x in range(ti.math.floor(pos.x - radius), ti.math.ceil(pos.x + radius)):
        for y in range(ti.math.floor(pos.y - radius), ti.math.ceil(pos.y + radius)):
            pixel = ti.math.vec2(x, y)
            dist = ti.math.distance(pixel, pos)
            pixels[x, y] += strength


@ti.kernel
def paint_point(pos: ti.math.vec2, size: ti.f32, cursor: ti.math.vec2, zone: ti.f32):
    radius = size
    distCursor = ti.math.distance(cursor, pos)
    if (distCursor <= zone):
        radius += (1 - distCursor / zone) * (0.9 - radius)

    for x in range(ti.math.floor(pos.x - zone), ti.math.ceil(pos.x + zone)):
        for y in range(ti.math.floor(pos.y - zone), ti.math.ceil(pos.y + zone)):
            pixel = ti.math.vec2(x, y)
            dist = ti.math.distance(pixel, pos)
            pixels[x, y] = white - (white - pixels[x, y]) * (1 - ti.math.pow(radius - 0.001, dist / strong))

            # pixels[x,y] = ti.math.vec3(1,1,1)*(1-ti.math.pow(radius-0.001, dist/strong))
            # pixels[x,y] *= (1-ti.math.pow(radius-0.001, dist/strong))


@ti.kernel
def paint_bg(color: ti.math.vec3):
    for x, y in pixels:
        rgb = color
        pixels[x, y] = rgb


def create_line(pointA, pointB, width, strength):
    n = (abs(pointB[0] - pointA[0]) + abs(pointB[1] - pointB[1]))
    n = math.floor(n)
    unitX = (pointB[0] - pointA[0]) / n
    unitY = (pointB[1] - pointA[1]) / n
    linePoints = ti.Vector.field(2, dtype=ti.f32, shape=n)
    for i in range(n):
        posX = pointA[0] + unitX * i
        posY = pointA[1] + unitY * i
        linePoints[i] = (posX, posY)
        paint_line_point(pos=linePoints[i], radius=width, strength=strength)


def main():
    window = ti.ui.Window('Window', (res, res), vsync=True)

    gui = window.get_gui()
    canvas = window.get_canvas()

    while window.running:
        paint_bg(color=black)
        cursor = ti.math.vec2(window.get_cursor_pos()) * res

        for n in range(vertices.shape[0]):
            paint_point(pos=vertices[n], size=0.7, cursor=cursor, zone=30)

        n = vertices.shape[0]
        create_line(pointA=(vertices[0][0], vertices[0][1]), pointB=(vertices[n - 1][0], vertices[n - 1][1]), width=0.1,
                    strength=0.2)

        canvas.set_image(pixels)
        window.show()


if __name__ == '__main__':
    main()
