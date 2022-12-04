import taichi as ti

# from linkage import Linkage
from .linkage import Linkage

windowSize = 768

ti.init(arch=ti.cuda)

driverColor = ti.math.vec3(ti.hex_to_rgb(0xd88c9a))
trackColor = ti.math.vec3(ti.hex_to_rgb(0x99c1b9))

purple = ti.math.vec3(ti.hex_to_rgb(0x8e7dbe))
green = ti.math.vec3(ti.hex_to_rgb(0x81B29A))
blue = ti.math.vec3(ti.hex_to_rgb(0x4f5d75))
yellow = ti.math.vec3(ti.hex_to_rgb(0xef8354))

trackList = ti.Vector.field(2, dtype=ti.f32, shape=1000)
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(windowSize, windowSize))
strong = windowSize * 0.001
black = ti.math.vec3(0, 0, 0)
white = ti.math.vec3(1, 1, 1)
red = ti.math.vec3(1, 0, 0)


@ti.func
def paint_line_point(pos: ti.math.vec2, radius: ti.f32, strength: ti.f32, color: ti.math.vec3):
    # call_time[0] += 1
    for x in range(int(ti.math.floor(pos.x - radius)), int(ti.math.ceil(pos.x + radius))):
        for y in range(int(ti.math.floor(pos.y - radius)), int(ti.math.ceil(pos.y + radius))):
            # call_time[0] += 1
            # rgb = ti.math.pow(0.2, strength * 0.5) * color * 0.1
            # rgb = strength * color * 0.1
            pixel = ti.math.vec2(x, y)
            dist = ti.math.distance(pixel, pos)

            rgb = (1 - ti.math.pow(radius - 0.001, dist / strong) * (strength * 6) * color)
            pixels[x, y] = white - (white - pixels[x, y]) * rgb

            # rgb = strength
            # pixels[x, y] = white - (white - pixels[x, y]) * rgb
            pass


@ti.func
def paint_point(pos: ti.math.vec2, size: ti.f32, cursor: ti.math.vec2, zone: ti.f32, strength: ti.f32,
                color: ti.math.vec3, notTrack: ti.u8):
    radius = size
    distCursor = ti.math.distance(cursor, pos)
    if distCursor <= zone and notTrack != 0:
        radius += (1 - distCursor / zone) * (0.9 - radius)
        strength *= 2

    for x in range(int(ti.math.floor(pos.x - zone)), int(ti.math.ceil(pos.x + zone))):
        for y in range(int(ti.math.floor(pos.y - zone)), int(ti.math.ceil(pos.y + zone))):
            pixel = ti.math.vec2(x, y)
            dist = ti.math.distance(pixel, pos)

            rgb = (1 - ti.math.pow(radius - 0.001, dist / strong) * (strength * 2) * color)
            pixels[x, y] = white - (white - pixels[x, y]) * rgb

            # pixels[x,y] += (1-rgb)

            # pixels[x,y] = ti.math.vec3(1,1,1)*(1-ti.math.pow(radius-0.001, dist/strong))
            # pixels[x,y] *= (1-ti.math.pow(radius-0.001, dist/strong))


@ti.kernel
def create_points(vertices: ti.template(), cursor: ti.math.vec2, tracked: ti.template(), driver: ti.i32):
    for n in range(vertices.shape[0]):
        pos = trans_pos(vertices[n].xy)
        if n == driver:
            paint_point(pos=pos, size=0.8, cursor=cursor, zone=30., strength=.8, color=red, notTrack=1)
        if tracked[n][0] != 0:
            paint_point(pos=pos, size=0.8, cursor=cursor, zone=30., strength=1., color=yellow, notTrack=0)
        else:
            paint_point(pos=pos, size=0.4, cursor=cursor, zone=30., strength=.6, color=purple, notTrack=1)


@ti.kernel
def paint_bg(color: ti.math.vec3, isPreview: ti.u8):
    for x, y in pixels:
        rgb = color
        if isPreview != 0:
            # pixels[x, y] *= 0.9
            # pixels[x,y] -= pixels[x,y]/100
            pixels[x, y] -= 0.03 - pixels[x, y] * 0.01
        else:
            pixels[x, y] = rgb


@ti.func
def trans_pos(pos: ti.math.vec2) -> ti.math.vec2:
    position = (pos + (30, 30)) * 15
    return position


@ti.kernel
def paint_line(vertices: ti.template(), indices: ti.template(), color: ti.math.vec3, cursor: ti.math.vec2):
    for i in range(indices.shape[0]):
        pointA = trans_pos(vertices[indices[i][0]].xy)
        pointB = trans_pos(vertices[indices[i][1]].xy)
        n = ti.math.distance(pointB, pointA)
        # n = (abs(pointB[0] - pointA[0]) + abs(pointB[1] - pointA[1]))
        n = ti.math.floor(n) + 1
        width = 0.5
        strength = 0.1
        unitX = (pointB[0] - pointA[0]) / n
        unitY = (pointB[1] - pointA[1]) / n

        for j in range(int(n)):
            posX = pointA[0] + unitX * j
            posY = pointA[1] + unitY * j
            # paint_point(ti.math.vec2(posX, posY), width, cursor, 30, strength, color, 0)
            paint_line_point(ti.math.vec2(posX, posY), radius=width, strength=strength, color=color)
            # paint_line_point(pos=(posX, posY), radius=width, strength=strength)


@ti.kernel
def paint_track(step: ti.i32, trackedPoints: ti.template(), cursor: ti.math.vec2):
    for n in ti.grouped(trackedPoints):
        pos = trans_pos(trackedPoints[n])
        now = step % 240
        nowA = now if now < 120 else 239 - now
        dist = abs(nowA - n[1] % 120) / 120
        # dist = min(abs(now - n[1]), (290-now-n[1]))/145
        size = (1 - dist) * 0.5
        strength = (1 - dist + 0.1) * 0.2

        if all(trackedPoints[n] != [0, 0]):
            # size = (1-(abs((step%145)-n[1]))/144)*0.4
            paint_point(pos=pos, size=size, cursor=cursor, zone=30., strength=strength, color=yellow, notTrack=1)


@ti.kernel
def get_tracked_points(vertices: ti.template(), isTracked: ti.template(), trackedPoints: ti.template(), steps: ti.i32):
    i = 0
    for n in isTracked:
        if isTracked[n][0] != 0:
            trackedPoints[i, steps] = vertices[n].xy
            i += 1


def show(linkage: Linkage):
    isRun = 1
    isPreview = 0

    window = ti.ui.Window("Leafall Linkage", (windowSize, windowSize))
    canvas = window.get_canvas()

    steps = 0
    step_diff = 1

    trackedPoints = ti.Vector.field(2, dtype=ti.f32, shape=(linkage.get_trackedNum(), 120))

    while window.running:
        linkage.substep(steps)
        vertices = linkage.get_vertices()
        indices = linkage.get_indices()
        isTracked = linkage.get_istracked()
        cursor = ti.math.vec2(window.get_cursor_pos()) * windowSize

        if steps < 120:
            get_tracked_points(vertices, isTracked, trackedPoints, steps)

        steps += step_diff

        paint_bg(black, isPreview)

        if window.get_event(ti.ui.PRESS):
            if window.event.key == ' ':  # space
                step_diff = 1 - step_diff
            if window.event.key == 'Return':  # 1
                if isPreview != 0:
                    isPreview = 0
                else:
                    isPreview = 1
                # isPreview = 1 if !isPreview else isPreview = 0

        driver = linkage.get_driver()

        create_points(vertices, cursor, isTracked, driver)

        if isPreview != 1:
            paint_line(vertices, indices, purple, cursor)

        paint_track(steps, trackedPoints=trackedPoints, cursor=cursor)

        canvas.set_image(pixels)
        window.show()
