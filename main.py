import taichi as ti

ti.init(arch=ti.cuda)

dt = 1e-3
substeps = int(1 / 60 // dt)

N = 2

indices = ti.Vector.field(2, dtype=ti.i32, shape=N * 2)
indices[0] = [0, 1]

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


def main():
    window = ti.ui.Window("Linkage 0", (768, 768))
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
        scene.lines(vertices, indices=indices, color=(0.28, 0.68, 0.99), width=5.0)

        canvas.scene(scene)
        window.show()


if __name__ == '__main__':
    main()
