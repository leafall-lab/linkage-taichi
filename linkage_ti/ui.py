import taichi as ti

# from linkage import Linkage
from .linkage import Linkage

windowSize = 768
strong = windowSize * 0.001

ti.init(arch=ti.cpu)

driverColor = ti.math.vec3(ti.hex_to_rgb(0xd88c9a))
trackColor = ti.math.vec3(ti.hex_to_rgb(0x99c1b9))

black = ti.math.vec3(0, 0, 0)
white = ti.math.vec3(1, 1, 1)
red = ti.math.vec3(1, 0, 0)
purple = ti.math.vec3(ti.hex_to_rgb(0x8e7dbe))
green = ti.math.vec3(ti.hex_to_rgb(0x81B29A))
blue = ti.math.vec3(ti.hex_to_rgb(0x4f5d75))
yellow = ti.math.vec3(ti.hex_to_rgb(0xef8354))

trackList = ti.Vector.field(2, dtype=ti.f32, shape=1000)
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(windowSize, windowSize))


@ti.func
def paint_line_point(pos: ti.math.vec2, radius: ti.f32, strength: ti.f32, color: ti.math.vec3):
    for x in range(int(ti.math.floor(pos.x - radius)), int(ti.math.ceil(pos.x + radius))):
        for y in range(int(ti.math.floor(pos.y - radius)), int(ti.math.ceil(pos.y + radius))):
            pixel = ti.math.vec2(x, y)
            dist = ti.math.distance(pixel, pos)

            rgb = (1 - ti.math.pow(radius - 0.001, dist / strong) * (strength * 6) * color)
            pixels[x, y] = white - (white - pixels[x, y]) * rgb


@ti.func
def paint_point(pos: ti.math.vec2, size: ti.f32, cursor: ti.math.vec2, zone: ti.f32, strength: ti.f32,
                color: ti.math.vec3, notTrack: ti.u8):
    radius = size
    distCursor = ti.math.distance(cursor, pos)
    if (distCursor <= zone and notTrack != 0):
        radius += (1 - distCursor / zone) * (0.9 - radius)
        strength *= 2

    for x in range(int(ti.math.floor(pos.x - zone)), int(ti.math.ceil(pos.x + zone))):
        for y in range(int(ti.math.floor(pos.y - zone)), int(ti.math.ceil(pos.y + zone))):
            pixel = ti.math.vec2(x, y)
            dist = ti.math.distance(pixel, pos)

            rgb = (1 - ti.math.pow(radius - 0.001, dist / strong) * (strength * 2) * color)
            pixels[x, y] = white - (white - pixels[x, y]) * rgb


@ti.kernel
def create_points(vertices: ti.template(), cursor: ti.math.vec2, tracked: ti.template(), driver: ti.i32,
                  driverColor: ti.math.vec3, trackColor: ti.math.vec3, lineColor: ti.math.vec3, trackedSize: ti.f32,
                  zoom: ti.f32, x: ti.f32,
                  y: ti.f32):
    for n in range(vertices.shape[0]):
        pos = trans_pos(vertices[n].xy, zoom, x, y)
        if (n == driver):
            paint_point(pos=pos, size=trackedSize, cursor=cursor, zone=30., strength=.8, color=driverColor, notTrack=1)
        if (tracked[n][0] != 0):
            paint_point(pos=pos, size=trackedSize, cursor=cursor, zone=30., strength=1., color=trackColor, notTrack=0)
        else:
            paint_point(pos=pos, size=0.4, cursor=cursor, zone=30., strength=.6, color=lineColor, notTrack=1)


@ti.kernel
def paint_bg(color: ti.math.vec3, isPreview: ti.u8):
    for x, y in pixels:
        rgb = color
        if (isPreview != 0):
            pixels[x, y] -= 0.03 - pixels[x, y] * 0.01
        else:
            pixels[x, y] = rgb


@ti.func
def trans_pos(pos: ti.math.vec2, zoom: ti.f32, x: ti.f32, y: ti.f32) -> ti.math.vec2:
    # position = (pos + (5, 10)) * 40
    # position = (pos + (15, 30)) * 20
    position = (pos + (x, y)) * zoom
    # position = (pos + (2, 4)) * 100
    return position


@ti.kernel
def paint_line(vertices: ti.template(), indices: ti.template(), color: ti.math.vec3, strength: ti.f32, zoom: ti.f32,
               x: ti.f32,
               y: ti.f32):
    for i in range(indices.shape[0]):
        pointA = trans_pos(vertices[indices[i][0]].xy, zoom, x, y)
        pointB = trans_pos(vertices[indices[i][1]].xy, zoom, x, y)
        n = ti.math.distance(pointB, pointA)
        # n = (abs(pointB[0] - pointA[0]) + abs(pointB[1] - pointA[1]))
        n = ti.math.floor(n) + 1
        width = 0.5
        unitX = (pointB[0] - pointA[0]) / n
        unitY = (pointB[1] - pointA[1]) / n

        for j in range(int(n)):
            posX = pointA[0] + unitX * j
            posY = pointA[1] + unitY * j
            # paint_point(ti.math.vec2(posX, posY), width, cursor, 30, strength, color, 0)
            paint_line_point(ti.math.vec2(posX, posY), radius=width, strength=strength, color=color)
            # paint_line_point(pos=(posX, posY), radius=width, strength=strength)


@ti.kernel
def paint_track(step: ti.i32, trackedPoints: ti.template(), cursor: ti.math.vec2, color: ti.math.vec3,
                trackedSize: ti.f32, zoom: ti.f32,
                x: ti.f32, y: ti.f32):
    for n in ti.grouped(trackedPoints):
        pos = trans_pos(trackedPoints[n], zoom, x, y)
        now = step % 240
        nowA = now if now < 120 else 239 - now
        dist = abs(nowA - n[1] % 120) / 120
        size = (1 - dist) * trackedSize
        strength = (1 - dist + 0.1) * 0.9

        if all(trackedPoints[n] != [0, 0]):
            paint_point(pos=pos, size=size, cursor=cursor, zone=30., strength=strength, color=color, notTrack=1)


@ti.kernel
def get_tracked_points(vertices: ti.template(), isTracked: ti.template(), trackedPoints: ti.template(), steps: ti.i32):
    i = 0
    for n in isTracked:
        if isTracked[n][0] != 0:
            trackedPoints[i, steps] = vertices[n].xy
            i += 1


def show(linkage: Linkage):
    isPreview = 0
    isPressing = 0

    window = ti.ui.Window("Leafall Linkage", (windowSize, windowSize))
    canvas = window.get_canvas()

    steps = 0
    step_diff = 1

    zoom = 20
    x = 10
    y = 15
    trackedSize = 0.7

    driverColor = ti.math.vec3(ti.hex_to_rgb(0xd88c9a))
    trackColor = ti.math.vec3(ti.hex_to_rgb(0x38a3a5))
    lineColor = ti.math.vec3(ti.hex_to_rgb(0x22577a))
    trackedPoints = ti.Vector.field(2, dtype=ti.f32, shape=(linkage.get_trackedNum(), 120))

    while window.running:
        linkage.substep(steps)
        steps += step_diff

        if window.get_event(ti.ui.PRESS):
            if window.event.key == ' ':  # space
                step_diff = 1 - step_diff
                isPressing = 0

            if window.event.key == 'Return':  # 1
                isPreview = 1 - isPreview

            if window.event.key == ti.ui.LMB:
                isPressing = 1

        if window.get_event(ti.ui.RELEASE):
            if window.event.key == ti.ui.LMB:
                isPressing = 0
                driverColor = ti.math.vec3(ti.hex_to_rgb(0xd88c9a))

        if window.is_pressed('z'):
            zoom += 1
        if window.is_pressed('x'):
            zoom -= 1
        if window.is_pressed(ti.ui.LEFT):
            x += 1
        if window.is_pressed(ti.ui.RIGHT):
            x -= 1
        if window.is_pressed(ti.ui.DOWN):
            y += 1
        if window.is_pressed(ti.ui.UP):
            y -= 1
        if window.is_pressed('n') and trackedSize < 0.99:
            trackedSize += 0.01
        if window.is_pressed('m') and trackedSize > 0.01:
            trackedSize -= 0.01

        vertices = linkage.get_vertices()
        indices = linkage.get_indices()
        isTracked = linkage.get_istracked()
        cursor = ti.math.vec2(window.get_cursor_pos()) * windowSize

        if steps < 120:
            get_tracked_points(vertices, isTracked, trackedPoints, steps)

        paint_bg(black, isPreview)
        driver = linkage.get_driver()

        create_points(vertices, cursor, isTracked, driver, driverColor, trackColor, lineColor, trackedSize, zoom, x, y)
        paint_track(steps, trackedPoints, cursor, trackColor, trackedSize, zoom, x, y)

        # paint_track(vertices, isTracked)
        if (isPreview != 1):
            paint_line(vertices, indices, lineColor, trackedSize / 2, zoom, x, y)

        if (isPressing == 1):
            driverColor = ti.hex_to_rgb(0xfca311)
            paint_bg(black, isPreview)
            create_points(vertices, cursor, isTracked, driver, driverColor, trackColor, lineColor, trackedSize, zoom, x,
                          y)
            paint_line(vertices, indices, lineColor, trackedSize / 2, zoom, x, y)
            paint_track(steps, trackedPoints, cursor, trackColor, trackedSize, zoom, x, y)

        canvas.set_image(pixels)
        window.show()
