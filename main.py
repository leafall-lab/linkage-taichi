import enum
import math
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
                [id1, r1, id2, r2, hint, [anti-hint-vertex]], anti-hint means the node's position should not equal on anti-hint-vertex
                this is more priority than "hint"

        Example::
            Vertex(VertexType.Fixed, [0, 0])
        """
        if tp == VertexType.Fixed:
            assert len(param) == 2  # todo: support 3-D position
        elif tp == VertexType.Driver:
            assert len(param) == 5
        elif tp == VertexType.Driven:
            assert len(param) == 5 or len(param) == 6

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
                id1, r1, id2, r2, hint = info.param[:5]
                x1, y1 = self.vertices[id1][0], self.vertices[id1][1]
                x2, y2 = self.vertices[id2][0], self.vertices[id2][1]
                x0, y0 = intersect_of_circle(x1, y1, r1, x2, y2, r2)[hint]
                if len(info.param) == 6:
                    anti_hint = info.param[5]
                    xh, yh = self.vertices[anti_hint][0], self.vertices[anti_hint][1]
                    if x0 - xh < 1e-5 and y0 - yh < 1e-5:
                        x0, y0 = intersect_of_circle(x1, y1, r1, x2, y2, r2)[1 - hint]
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
        VertexInfo(VertexType.Fixed, [3.0, -7.5]),
        VertexInfo(VertexType.Fixed, [3.0, -4.5]),
        VertexInfo(VertexType.Driver, [3.0, -4.5, 3.0, 0.8, 2.25]),
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
        VertexInfo(VertexType.Fixed, [3.0, -7.5]),  # 0
        VertexInfo(VertexType.Fixed, [3.0, -4.5]),  # 1
        VertexInfo(VertexType.Driver, [3.0, -4.5, 3.0, 0.8, 2.25]),  # 2, driver
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


def Adder() -> Linkage:
    basic: float = 3.2

    adder_len = 3.2

    info = [
        # x-axis
        VertexInfo(VertexType.Fixed, [3.0, -7.5]),  # 0
        VertexInfo(VertexType.Fixed, [3.0, -4.5]),  # 1
        VertexInfo(VertexType.Driver, [3.0, -4.5, 3.0, 0.8, 2.25]),  # 2, driver
        VertexInfo(VertexType.Driven, [0, 7.0, 2, 2.0, 0]),  # 3
        VertexInfo(VertexType.Driven, [0, 7.0, 2, 2.0, 1]),  # 4
        VertexInfo(VertexType.Driven, [3, 2.0, 4, 2.0, 0]),  # 5, Peaucellier straight line, (0,0) <-> (6,0), x-axis
        # y-axis
        VertexInfo(VertexType.Fixed, [0.0, 0.0]),  # 6, origin
        VertexInfo(VertexType.Driven, [5, basic, 6, basic, 0]),  # 7
        VertexInfo(VertexType.Driven, [5, basic, 6, basic, 1]),  # 8
        VertexInfo(VertexType.Driven, [6, basic, 7, basic * math.sqrt(2), 0]),  # 9
        VertexInfo(VertexType.Driven, [6, basic, 8, basic * math.sqrt(2), 0]),  # 10
        VertexInfo(VertexType.Driven, [9, basic, 10, basic, 1]),  # 11, y-axis
        # add x+y
        VertexInfo(VertexType.Driven, [6, adder_len, 5, adder_len, 1]),  # 12
        VertexInfo(VertexType.Driven, [6, adder_len, 11, adder_len, 1]),  # 13
        VertexInfo(VertexType.Driven, [12, adder_len, 13, adder_len, 1]),  # 14
        VertexInfo(VertexType.Driven, [5, adder_len, 14, adder_len, 1, 12]),  # 15, not equal on 12
        VertexInfo(VertexType.Driven, [14, adder_len, 11, adder_len, 1, 13]),  # 16, not equal on 13
        VertexInfo(VertexType.Driven, [15, adder_len, 16, adder_len, 1]),  # 17, x-y=0
    ]
    extra_lines = [
        [1, 2], [6, 5], [6, 11], [6, 17]
    ]

    # blue = (0.28, 0.68, 0.99)
    # green = (0.68, 0.99, 0.28)
    # red = (0.99, 0.28, 0.68)
    blue = (0, 0, 0)
    green = (0, 0, 0)
    red = (0.99, 0.28, 0.68)
    yellow = (0.99, 0.99, 0.28)
    colors = [
        blue, blue, blue, blue, blue,
        red,
        green, green, green, green, green,
        red,
        yellow, yellow, yellow, yellow, yellow, red
    ]

    return Linkage(info, extra_lines, colors)


def Multiplier(multiple: float = 0.5) -> Linkage:
    basic: float = 3.2

    info = [
        # x-axis
        VertexInfo(VertexType.Fixed, [3.0, -7.5]),  # 0
        VertexInfo(VertexType.Fixed, [3.0, -4.5]),  # 1
        VertexInfo(VertexType.Driver, [3.0, -4.5, 3.0, 0.8, 2.25]),  # 2, driver
        VertexInfo(VertexType.Driven, [0, 7.0, 2, 2.0, 0]),  # 3
        VertexInfo(VertexType.Driven, [0, 7.0, 2, 2.0, 1]),  # 4
        VertexInfo(VertexType.Driven, [3, 2.0, 4, 2.0, 0]),  # 5, Peaucellier straight line, (0,0) <-> (6,0), x-axis
        VertexInfo(VertexType.Fixed, [0.0, 0.0]),  # 6, origin
        # multiplier
        VertexInfo(VertexType.Driven, [5, basic, 6, basic, 1]),  # 7
        VertexInfo(VertexType.Driven, [6, multiple * basic, 7, (0.0001 + abs(multiple - 1)) * basic, 1]),  # 8
        VertexInfo(VertexType.Driven, [5, (0.0001 + abs(multiple - 1)) * basic, 8, basic, 0 if multiple < 1 else 1]),
        # 9
        VertexInfo(VertexType.Driven, [8, multiple * basic, 9, (0.0001 + abs(multiple - 1)) * basic, 1]),  # 10
    ]
    extra_lines = [
        [1, 2]
    ]

    # blue = (0.28, 0.68, 0.99)
    # green = (0.68, 0.99, 0.28)
    # red = (0.99, 0.28, 0.68)
    # blue = (0, 0, 0)
    # green = (0, 0, 0)
    # red = (0.99, 0.28, 0.68)
    # yellow = (0.99, 0.99, 0.28)
    # colors = [
    #     # blue, blue, blue, blue, blue,
    #     # red,
    #     # green, green, green, green, green,
    #     # red,
    #     # yellow, yellow, yellow, yellow, yellow, red
    # ]

    return Linkage(info, extra_lines)


class LinkageBuilder:
    def __init__(self, global_color_hint: Tuple[float, float, float] = None):
        self.infos: List[VertexInfo] = []
        self.extra_lines: List[List[int]] = []
        self.colors: List[Tuple[float, float, float]] = []
        self.global_color_hint = global_color_hint if global_color_hint is not None else (0.28, 0.68, 0.99)

    def register_color(self, old_n: int, color_hint: Tuple[float, float, float]):
        color = color_hint if color_hint is not None else self.global_color_hint

        if old_n < self.vertices():
            self.colors.extend([color] * (self.vertices() - old_n))

    # add Peaucellier straight line
    # need: nothing
    # return: id of x
    def add_straight_line(self, color_hint: Tuple[float, float, float] = None) -> int:
        n: int = self.vertices()

        self.infos.extend([
            VertexInfo(VertexType.Fixed, [3.0, -7.5]),  # +0
            VertexInfo(VertexType.Fixed, [3.0, -4.5]),  # +1
            VertexInfo(VertexType.Driver, [3.0, -4.5, 3.0, 0.8, 2.2]),  # +2
            VertexInfo(VertexType.Driven, [n, 7.0, n + 2, 2.0, 0]),  # +3
            VertexInfo(VertexType.Driven, [n, 7.0, n + 2, 2.0, 1]),  # +4
            VertexInfo(VertexType.Driven, [n + 3, 2.0, n + 4, 2.0, 0]),  # +5, x-axis
        ])
        self.extra_lines.append([n + 1, n + 2])

        self.register_color(n, color_hint)
        return self.vertices() - 1

    # add fixed point
    # need: nothing
    # return: id of point
    def add_fixed(self, x: float = 0.0, y: float = 0.0, color_hint: Tuple[float, float, float] = None) -> int:
        self.infos.extend([VertexInfo(VertexType.Fixed, [x, y])])  # +0

        self.register_color(self.vertices() - 1, color_hint)
        return self.vertices() - 1

    # add axes from straight line
    # need: id of origin and x
    # return: id of y
    def add_axes(self, o: int, x: int, color_hint: Tuple[float, float, float] = None) -> int:
        n: int = self.vertices()

        basic = 3.2

        self.infos.extend([
            VertexInfo(VertexType.Driven, [o, basic, x, basic, 1]),  # +0
            VertexInfo(VertexType.Driven, [o, basic, x, basic, 0]),  # +1
            VertexInfo(VertexType.Driven, [o, basic, n + 0, basic * math.sqrt(2), 0]),  # +2
            VertexInfo(VertexType.Driven, [o, basic, n + 1, basic * math.sqrt(2), 0]),  # +3
            VertexInfo(VertexType.Driven, [n + 2, basic, n + 3, basic, 1]),  # +4, y-axis
        ])

        self.register_color(n, color_hint)
        return self.vertices() - 1

    # add zoomer (constant multiplier)
    # need: id of origin, zoomee, multiple
    # return: id of result
    def add_zoomer(self, o: int, x: int, multi: float = 2, color_hint: Tuple[float, float, float] = None) -> int:
        n: int = self.vertices()
        basic = 3.2

        shorter = (0.0001 + abs(multi - 1)) * basic

        self.infos.extend([
            VertexInfo(VertexType.Driven, [x, basic, o, basic, 1]),  # +0
            VertexInfo(VertexType.Driven, [o, multi * basic, n + 0, shorter, 1]),  # +1
            VertexInfo(VertexType.Driven, [x, shorter, n + 1, basic, 0 if multi < 1 else 1]),  # +2
            VertexInfo(VertexType.Driven, [n + 1, multi * basic, n + 2, shorter, 1]),  # +3
        ])

        self.register_color(n, color_hint)
        return self.vertices() - 1

    # add vector adder
    # need: id of origin, op1, op2
    # return id of op1+op2
    def add_adder(self, o: int, a: int, b: int, color_hint: Tuple[float, float, float] = None) -> int:
        n: int = self.vertices()
        basic = 6.4
        # print("n=", n, " o=", o, " a=", a, " b=", b)

        self.infos.extend([
            VertexInfo(VertexType.Driven, [o, basic, a, basic, 1]),  # +0
            VertexInfo(VertexType.Driven, [o, basic, b, basic, 1]),  # +1
            VertexInfo(VertexType.Driven, [n + 0, basic, n + 1, basic, 1]),  # +2
            VertexInfo(VertexType.Driven, [a, basic, n + 2, basic, 1, n + 0]),  # +3, not equal to +0
            VertexInfo(VertexType.Driven, [n + 2, basic, b, basic, 1, n + 1]),  # +4, not equal to +1
            VertexInfo(VertexType.Driven, [n + 3, basic, n + 4, basic, 1]),  # +5
        ])

        self.register_color(n, color_hint)
        return self.vertices() - 1

    # add a mover (constant adder)
    # need: id of x, (dx, dy)
    # return id of moved point
    # TODO: refine direction hint logic of +4
    def add_mover(self, x: int, dx: float, dy: float, color_hint: Tuple[float, float, float] = None) -> int:
        n: int = self.vertices()
        basic = 6.4

        # x0 = random.uniform(-10, 10)
        # y0 = random.uniform(-10, 10)
        x0 = -5
        y0 = 6

        d = math.sqrt(dx * dx + dy * dy)

        self.infos.extend([
            VertexInfo(VertexType.Fixed, [x0, y0]),  # +0
            VertexInfo(VertexType.Fixed, [x0 + dx, y0 + dy]),  # +1
            VertexInfo(VertexType.Driven, [x, basic, n + 0, basic, 0]),  # +2
            VertexInfo(VertexType.Driven, [n + 1, basic, n + 2, d, 0]),  # +3
            VertexInfo(VertexType.Driven, [n + 3, basic, x, d, 1]),  # +4
        ])
        self.add_extra_lines([[n + 0, n + 1]])

        self.register_color(n, color_hint)
        return self.vertices() - 1

    # add an inverter
    # need: id of o and x
    # return: id of inverted index `t`
    # ot * ox = (a^2 - b^2)
    def add_inverter(self, o: int, x: int, color_hint: Tuple[float, float, float] = None) -> int:
        n: int = self.vertices()

        basic = 6.4
        tx = 6

        self.infos.extend([
            VertexInfo(VertexType.Driven, [o, basic, x, tx, 0]),  # n
            VertexInfo(VertexType.Driven, [o, basic, x, tx, 1]),  # n+1
            VertexInfo(VertexType.Driven, [n, tx, n + 1, tx, 0, x]),  # n+2
        ])
        # self.extra_lines.append([n + 1, n + 2])

        self.register_color(n, color_hint)
        return self.vertices() - 1

    # add a square ox^2
    # def add_square(self):

    def vertices(self):
        return len(self.infos)

    def get_linkage(self):
        return Linkage(self.infos, self.extra_lines, self.colors)

    # set p as the traced point (config it's color), and return the linkage
    def set_color(self, p: int, color: Tuple[float, float, float]):
        print("set_color", p, color)
        print("self.colors", len(self.colors))
        self.colors[p] = color

    def add_extra_lines(self, lines: List[List[int]]):
        self.extra_lines.extend(lines)


def YEqualKx(k: float = 2.0) -> Linkage:
    builder = LinkageBuilder()
    x = builder.add_straight_line()
    o = builder.add_fixed()
    y = builder.add_axes(o, x)
    y2 = builder.add_zoomer(o, y, k)
    p = builder.add_adder(o, x, y2)

    builder.set_color(p, (1.0, 0.0, 0.0))
    builder.add_extra_lines([[o, x], [o, y], [o, p]])

    return builder.get_linkage()


def main():
    ti.init(arch=ti.cuda)

    dt = 0.01  # <= 0.01, 0.01 is slowest.
    substeps = int(1 / 100 // dt)

    # # linkage = GrashofFourBarLinkage(3)
    # name = sys.argv[1] if len(sys.argv) > 1 else "Multiplier"
    # linkage = eval(name + "()")

    # linkage = YEqualKx(0.5)
    builder = LinkageBuilder()
    o = builder.add_fixed()
    x = builder.add_straight_line()
    builder.add_extra_lines([[o, x]])
    p = builder.add_inverter(o, x)
    builder.set_color(p, (1.0, 0.0, 0.0))

    linkage = builder.get_linkage()

    # result_dir = "/Users/lf/llaf/linkage-tc/results"
    # video_manager = ti.tools.VideoManager(output_dir=result_dir, framerate=24, automatic_build=False)

    window = ti.ui.Window("Leafall Linkage", (768, 768))
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(0, 0, 30)
    camera.lookat(0, 0, 0)

    current_t = 0.0
    steps = 0

    step_diff = 1

    while True:
        for i in range(substeps):
            linkage.substep(steps)
            steps += step_diff

            current_t += dt
        # print(linkage.get_vertices())

        if window.get_event(ti.ui.PRESS):
            if window.event.key == ' ':  # space
                step_diff = 1 - step_diff

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
