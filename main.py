import enum
import math
import sys
from typing import List, Tuple

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
            for Driver vertex, param contains center of circle, radius, initial angle, max angle  [x0, y0, r, theta0, theta1]
            for Driven vertex, param contains double (vertex id, radius) and a direction hint(0/1)
                [id1, r1, id2, r2, hint]

        Example::
            Vertex(VertexType.Fixed, [0, 0])
        """
        if tp == VertexType.Fixed:
            assert len(param) == 2  # todo: support 3-D position
        elif tp == VertexType.Driver:
            assert len(param) == 5
        elif tp == VertexType.Driven:
            assert len(param) == 5

        self.tp = tp
        self.param = param
        # self.step: int = 0


class Linkage:
    def __init__(self, vertex_infos: List[VertexInfo], lines: List[List[int]] = None,
                 colors: List[Tuple[float, float, float]] = None):
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

        if colors is not None:
            self.colors = ti.Vector.field(3, dtype=ti.f32, shape=len(colors))
            for i in range(len(colors)):
                self.colors[i] = colors[i]

    # needn't topo sort, we assume that small-id vertex is never rely on large-id vertex
    def substep(self, step: int):
        for i in range(self.N):
            info = self.vertex_infos[i]
            if info.tp == VertexType.Fixed:
                self.vertices[i] = [info.param[0], info.param[1], 0]
            elif info.tp == VertexType.Driver:
                cycle = info.param[4] - info.param[3]
                # theta = step * 0.01 % cycle + info.param[3]  # cycle
                theta = cycle - abs(cycle - step * 0.01 % (cycle * 2)) + info.param[3]  # wander
                self.vertices[i] = [info.param[0] + info.param[2] * math.cos(theta),
                                    info.param[1] + info.param[2] * math.sin(theta), 0]
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

    def get_colors(self):
        return self.colors if hasattr(self, 'colors') else None

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
    a3 = a2 - h * (y2 - y1) / d
    b3 = b2 + h * (x2 - x1) / d
    a4 = a2 + h * (y2 - y1) / d
    b4 = b2 - h * (x2 - x1) / d

    return [a3, b3], [a4, b4]


def linkage0() -> Linkage:
    info = [
        VertexInfo(VertexType.Fixed, [0, 0]),
        VertexInfo(VertexType.Driver, [0, 0, 5, 0, math.pi * 2]),
    ]
    extra_lines = [[0, 1]]
    return Linkage(info, extra_lines)


def linkage1() -> Linkage:
    info = [
        VertexInfo(VertexType.Fixed, [0, 0]),
        VertexInfo(VertexType.Driver, [0, 0, 5, 0, math.pi * 2]),
        VertexInfo(VertexType.Driven, [0, 3.5, 1, 3.5, 0]),
    ]
    return Linkage(info)


def GrashofFourBarLinkage(radius: float = 1.0) -> Linkage:
    info = [
        VertexInfo(VertexType.Fixed, [0.0, 0.0]),
        VertexInfo(VertexType.Fixed, [5.0, 0.0]),
        VertexInfo(VertexType.Driver, [0.0, 0.0, radius, 0.0, math.pi * 2]),
        VertexInfo(VertexType.Driven, [1, 7.0, 2, 6.0, 1]),
    ]
    extra_lines = [[0, 1], [0, 2]]
    return Linkage(info, extra_lines)


def PeaucellierStraightLinkage() -> Linkage:
    info = [
        VertexInfo(VertexType.Fixed, [3.0, 7.5]),
        VertexInfo(VertexType.Fixed, [3.0, 4.5]),
        VertexInfo(VertexType.Driver, [3.0, 4.5, 3.0, -2.25, -0.8]),
        VertexInfo(VertexType.Driven, [0, 7.0, 2, 2.0, 0]),
        VertexInfo(VertexType.Driven, [0, 7.0, 2, 2.0, 1]),
        VertexInfo(VertexType.Driven, [3, 2.0, 4, 2.0, 0]),
    ]
    extra_lines = [
        [1, 2]
    ]
    return Linkage(info, extra_lines)


def Axes() -> Linkage:
    basic: float = 3.2

    info = [
        VertexInfo(VertexType.Fixed, [3.0, 7.5]),  # 0
        VertexInfo(VertexType.Fixed, [3.0, 4.5]),  # 1
        VertexInfo(VertexType.Driver, [3.0, 4.5, 3.0, -2.25, -0.8]),  # 2, driver
        VertexInfo(VertexType.Driven, [0, 7.0, 2, 2.0, 0]),  # 3
        VertexInfo(VertexType.Driven, [0, 7.0, 2, 2.0, 1]),  # 4
        VertexInfo(VertexType.Driven, [3, 2.0, 4, 2.0, 0]),  # 5, Peaucellier straight line, (0,0) <-> (6,0), x-axis
        VertexInfo(VertexType.Fixed, [0.0, 0.0]),  # 6, origin
        VertexInfo(VertexType.Driven, [5, basic, 6, basic, 0]),  # 7
        VertexInfo(VertexType.Driven, [5, basic, 6, basic, 1]),  # 8
        VertexInfo(VertexType.Driven, [6, basic, 7, basic * math.sqrt(2), 0]),  # 9
        VertexInfo(VertexType.Driven, [6, basic, 8, basic * math.sqrt(2), 0]),  # 10
        VertexInfo(VertexType.Driven, [9, basic, 10, basic, 1]),  # 11, y-axis
    ]
    extra_lines = [
        [1, 2], [6, 5], [6, 11]
    ]

    blue = (0.28, 0.68, 0.99)
    green = (0.68, 0.99, 0.28)
    red = (0.99, 0.28, 0.68)
    # colors = [
    #     blue, blue, blue,
    #     green, green, green, green, green,
    #     blue, green, green
    # ]
    colors = [
        blue, blue, blue, blue, blue,
        red,
        green, green, green, green, green,
        red,
    ]

    return Linkage(info, extra_lines, colors)


def main():
    ti.init(arch=ti.cuda)

    dt = 0.01  # <= 0.01, 0.01 is slowest.
    substeps = int(1 / 100 // dt)

    # linkage = GrashofFourBarLinkage(3)
    name = sys.argv[1] if len(sys.argv) > 1 else "Axes"
    linkage = eval(name + "()")

    # result_dir = "/Users/lf/llaf/linkage-tc/results"
    # video_manager = ti.tools.VideoManager(output_dir=result_dir, framerate=24, automatic_build=False)

    window = ti.ui.Window("Leafall Linkage", (768, 768))
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(0, 0, 20)
    camera.lookat(0, 0, 0)

    current_t = 0.0
    steps = 0

    while True:
        for i in range(substeps):
            linkage.substep(steps)
            steps += 1

            current_t += dt
        # print(linkage.get_vertices())

        camera.track_user_inputs(window, movement_speed=0.05, hold_key=ti.ui.LMB)
        scene.set_camera(camera)
        scene.ambient_light((0.8, 0.8, 0.8))
        scene.point_light(pos=(0, 0, 20), color=(1, 1, 1))

        scene.particles(linkage.get_vertices(), color=(0.68, 0.26, 0.19), radius=0.1)
        scene.lines(linkage.get_vertices(), indices=linkage.get_indices(), color=(0.28, 0.68, 0.99), width=5.0,
                    vertex_count=linkage.get_even_n(), per_vertex_color=linkage.get_colors())

        canvas.scene(scene)
        # video_manager.write_frame(window.get_image_buffer_as_numpy())

        window.show()

    # video_manager.make_video(gif=True, mp4=False)


if __name__ == '__main__':
    main()
