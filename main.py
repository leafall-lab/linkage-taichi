import math

import taichi as ti

ti.init(arch=ti.cuda)

dt = 1e-3
substeps = int(1 / 60 // dt)

N = 3

indices = ti.Vector.field(2, dtype=ti.i32, shape=N * 2)
indices[0] = [0, 2]
indices[1] = [1, 2]
# indices[0] = [0, 1]

# 0: fixed point
# 1: driver point
# 2: driven point

fixed_point = ti.Vector.field(3, dtype=ti.f32, shape=1)
fixed_point[0] = [0.0, 0.0, 0.0]

x = ti.Vector.field(3, dtype=float, shape=1)
x[0] = [1.0, 0.0, 0.0]
v = ti.Vector.field(3, dtype=float, shape=1)

vertices = ti.Vector.field(3, dtype=ti.f32, shape=N)


def substep():
    v[0] = [-x[0][1], x[0][0], 0.0]
    x[0] += v[0] * dt


def update_vertices():
    vertices[0] = fixed_point[0]
    vertices[1] = x[0]

    inter = intersect_of_circle(fixed_point[0][0], fixed_point[0][1], 0.7, x[0][0], x[0][1], 0.7)
    vertices[2] = [inter[0], inter[1], 0.0]


def intersect_of_circle(x1, y1, r1, x2, y2, r2):
    d = math.sqrt((abs(x1 - x2)) ** 2 + (abs(y1 - y2)) ** 2)
    # if d > (r1 + r2) or d < (abs(r1 - r2)):
    #     print("Two circles have no intersection")
    #     return
    # elif d == 0 and r1 == r2:
    #     print("Two circles have same center!")
    #     return
    # else:
    A = (r1 ** 2 - r2 ** 2 + d ** 2) / (2 * d)
    h = math.sqrt(r1 ** 2 - A ** 2)

    a2 = x1 + A * (x2 - x1) / d
    b2 = y1 + A * (y2 - y1) / d
    a3 = round(a2 - h * (y2 - y1) / d, 2)
    b3 = round(b2 + h * (x2 - x1) / d, 2)
    # a4 = round(a2 + h * (y2 - y1) / d, 2)
    # b4 = round(b2 - h * (x2 - x1) / d, 2)
    #
    # d1 = np.array((a3, b3, 0.))
    # d2 = np.array((a4, b4, 0.))
    # print(d1, d2)
    return a3, b3


def main():
    window = ti.ui.Window("Linkage 1", (768, 768))
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(0, 0, 5)
    camera.lookat(0, 0, 0)

    current_t = 0.0

    while window.running:
        for i in range(substeps):
            substep()
            current_t += dt
        update_vertices()

        camera.track_user_inputs(window, movement_speed=0.05, hold_key=ti.ui.LMB)
        scene.set_camera(camera)
        scene.ambient_light((1, 1, 1))
        scene.point_light(pos=(0.5, 1.5, 1.5), color=(0.1, 0.1, 0.1))

        scene.particles(vertices, color=(0.68, 0.26, 0.19), radius=0.1)
        scene.lines(vertices, indices=indices, color=(0.28, 0.68, 0.99), width=5.0, vertex_count=(N + 1) // 2)

        canvas.scene(scene)
        window.show()


if __name__ == '__main__':
    main()
