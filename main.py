import enum
import math
from typing import List

import taichi as ti


@enum.unique
class VertexType(enum.Enum):
    Fixed = 0
    Driver = 1
    Driven = 2


class VertexInfo:
    """vertex information class for linkage system"""

    def __init__(self, tp: VertexType, param: List[float]):
        """ init vertex

        :param tp: vertex type
        :param param: some parameters for vertex,
            for Fixed vertex, param contains position [x, y]
            for Driver vertex, param contains initial position and circle central position [x0, y0, r]
            for Driven vertex, param contains double (vertex id, radius) and a direction hint(0/1)
                [id1, r1, id2, r2, hint]

        Example::
            Vertex(VertexType.Fixed, [0, 0])
        """
        if tp == VertexType.Fixed:
            assert len(param) == 2  # todo: support 3-D position
        elif tp == VertexType.Driver:
            assert len(param) == 3  # todo: support initial position
        elif tp == VertexType.Driven:
            assert len(param) == 5

        self.tp = tp
        self.param = param
        # self.step: int = 0


class Linkage:
    def __init__(self, vertex_infos: List[VertexInfo], lines: List[List[int]] = None):
        self.N: int = len(vertex_infos)
        self.vertex_infos = vertex_infos
        self.vertices = ti.Vector.field(3, dtype=ti.f32, shape=self.N)

        lines = lines or []
        for i in range(self.N):
            if self.vertex_infos[i].tp == VertexType.Driven:
                lines.append([i, self.vertex_infos[i].param[0]])
                lines.append([i, self.vertex_infos[i].param[2]])
        self.line_indices = ti.Vector.field(2, dtype=ti.i32, shape=len(lines))
        for i in range(len(lines)):
            self.line_indices[i] = lines[i]

    # needn't topo sort, we assume that small-id vertex is never rely on large-id vertex
    def substep(self, step: int):
        for i in range(self.N):
            info = self.vertex_infos[i]
            if info.tp == VertexType.Fixed:
                self.vertices[i] = [info.param[0], info.param[1], 0]
            elif info.tp == VertexType.Driver:
                self.vertices[i] = [info.param[0] + info.param[2] * math.cos(step * 0.01),
                                    info.param[1] + info.param[2] * math.sin(step * 0.01), 0]
            elif self.vertex_infos[i].tp == VertexType.Driven:
                id1, r1, id2, r2, hint = info.param
                x1, y1 = self.vertices[id1][0], self.vertices[id1][1]
                x2, y2 = self.vertices[id2][0], self.vertices[id2][1]
                x0, y0 = intersect_of_circle(x1, y1, r1, x2, y2, r2)[hint]
                self.vertices[i] = [x0, y0, 0]

    def get_vertices(self):
        return self.vertices

    def get_indices(self):
        return self.line_indices

    def get_even_n(self):
        return (self.N - 1) // 2 * 2


def intersect_of_circle(x1, y1, r1, x2, y2, r2):
    d = math.sqrt((abs(x1 - x2)) ** 2 + (abs(y1 - y2)) ** 2)
    if d > (r1 + r2) or d < (abs(r1 - r2)):
        print("Two circles have no intersection", x1, y1, r1, x2, y2, r2)
        return
    elif d == 0 and r1 == r2:
        print("Two circles have same center!")
        return

    A = (r1 ** 2 - r2 ** 2 + d ** 2) / (2 * d)
    h = math.sqrt(r1 ** 2 - A ** 2)

    a2 = x1 + A * (x2 - x1) / d
    b2 = y1 + A * (y2 - y1) / d
    a3 = round(a2 - h * (y2 - y1) / d, 2)
    b3 = round(b2 + h * (x2 - x1) / d, 2)
    a4 = round(a2 + h * (y2 - y1) / d, 2)
    b4 = round(b2 - h * (x2 - x1) / d, 2)

    return [a3, b3], [a4, b4]


def main():
    ti.init(arch=ti.cuda)

    dt = 0.0075
    substeps = int(1 / 60 // dt)

    info = [
        VertexInfo(VertexType.Fixed, [0.0, 0.0]),
        VertexInfo(VertexType.Fixed, [5.0, 0.0]),
        VertexInfo(VertexType.Driver, [0.0, 0.0, 1.0]),
        VertexInfo(VertexType.Driven, [1, 7.0, 2, 6.0, 1]),
    ]
    extra_lines = [
        [0, 1], [0, 2]
    ]

    linkage = Linkage(info, extra_lines)

    window = ti.ui.Window("Grashof’s Four-Bar Linkage", (768, 768))
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(0, 0, 20)
    camera.lookat(0, 0, 0)

    current_t = 0.0
    steps = 0
    while window.running:
        for i in range(substeps):
            linkage.substep(steps)
            steps += 1

            current_t += dt

        camera.track_user_inputs(window, movement_speed=0.05, hold_key=ti.ui.LMB)
        scene.set_camera(camera)
        scene.ambient_light((1, 1, 1))
        scene.point_light(pos=(0.5, 1.5, 1.5), color=(0.1, 0.1, 0.1))

        scene.particles(linkage.get_vertices(), color=(0.68, 0.26, 0.19), radius=0.5)
        scene.lines(linkage.get_vertices(), indices=linkage.get_indices(), color=(0.28, 0.68, 0.99), width=5.0,
                    vertex_count=linkage.get_even_n())

        canvas.scene(scene)
        window.show()


if __name__ == '__main__':
    main()
