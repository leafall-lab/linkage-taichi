import enum
import math
import random
from typing import List, Tuple

import taichi as ti

ti.init(arch=ti.cuda)
res = 768
strong = res * 0.001
black = ti.math.vec3(0, 0, 0)
white = ti.math.vec3(1, 1, 1)
red = ti.math.vec3(1, 0, 0)

driverColor = ti.math.vec3(ti.hex_to_rgb(0xd88c9a))
trackColor = ti.math.vec3(ti.hex_to_rgb(0x99c1b9))

purple = ti.math.vec3(ti.hex_to_rgb(0x8e7dbe))
green = ti.math.vec3(ti.hex_to_rgb(0x81B29A))
blue = ti.math.vec3(ti.hex_to_rgb(0x4f5d75))
yellow = ti.math.vec3(ti.hex_to_rgb(0xef8354))

pixels = ti.Vector.field(3, dtype=ti.f32, shape=(res, res))
trackList = ti.Vector.field(2, dtype=ti.f32, shape=1000)


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
                 colors: List[Tuple[float, float, float]] = None, tracked: List[int] = None, driver: int = -1):
        self.N: int = len(vertex_infos)
        self.vertex_infos = vertex_infos
        self.vertices = ti.Vector.field(3, dtype=ti.f32, shape=self.N)
        self.driver = driver
        self.trackedNum = 0

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

        if tracked is not None:
            self.trackedNum = len(tracked)
            self.tracked = ti.Vector.field(1, dtype=ti.u8, shape=self.N)
            for i in range(len(tracked)):
                self.tracked[tracked[i]][0] = 1

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
                x3, y3 = intersect_of_circle(x1, y1, r1, x2, y2, r2)[hint]
                # if i == 5:
                #     print(x3)  # 0.6799779794256628, 5.320021789745519

                if len(info.param) == 6:  # if it can't form a Parallelogram, use the other intersection
                    anti_hint = info.param[5]
                    x0, y0 = self.vertices[anti_hint][0], self.vertices[anti_hint][1]

                    x3d, y3d = intersect_of_circle(x1, y1, r1, x2, y2, r2)[1 - hint]
                    diff1 = abs((y1 - y0) * (x3 - x2) - (y3 - y2) * (x1 - x0))
                    diffd = abs((y1 - y0) * (x3d - x2) - (y3d - y2) * (x1 - x0))
                    if diffd + 1e-3 < diff1:
                        x3, y3 = x3d, y3d
                        info.param[4] = 1 - info.param[4]
                self.vertices[i] = [x3, y3, 0]

    def get_vertices(self):
        return self.vertices

    def get_indices(self):
        return self.line_indices

    def get_colors(self):
        return self.colors if hasattr(self, 'colors') else None

    def get_even_n(self):
        return (self.N - 1) // 2 * 2

    def get_istracked(self):
        return self.tracked

    def get_trackedNum(self):
        return self.trackedNum

    def get_driver(self):
        return self.driver


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
    return Linkage(info, extra_lines, None, None, 1)


def linkage1() -> Linkage:
    info = [
        VertexInfo(VertexType.Fixed, [0, 0]),
        VertexInfo(VertexType.Driver, [0, 0, 5, 0, math.pi * 2]),
        VertexInfo(VertexType.Driven, [0, 3.5, 1, 3.5, 0]),
    ]
    return Linkage(info, None, None, [2], 1)


def GrashofFourBarLinkage(radius: float = 1.0) -> Linkage:
    info = [
        VertexInfo(VertexType.Fixed, [0.0, 0.0]),
        VertexInfo(VertexType.Fixed, [5.0, 0.0]),
        VertexInfo(VertexType.Driver, [0.0, 0.0, radius, 0.0, math.pi * 2]),
        VertexInfo(VertexType.Driven, [1, 7.0, 2, 6.0, 1]),
    ]
    extra_lines = [[0, 1], [0, 2]]
    return Linkage(info, extra_lines, None, [3], 2)


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
    return Linkage(info, extra_lines, None, [5], 2)


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

    return Linkage(info, extra_lines, colors, [5, 11], 2)


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

    return Linkage(info, extra_lines, colors, [5, 11, 17], 2)


def Zoomer(multiple: float = 0.5) -> Linkage:
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
        # zoomer
        VertexInfo(VertexType.Driven, [5, basic, 6, basic, 1]),  # 7
        VertexInfo(VertexType.Driven, [6, multiple * basic, 7, (0.0001 + abs(multiple - 1)) * basic, 1]),  # 8
        VertexInfo(VertexType.Driven, [5, (0.0001 + abs(multiple - 1)) * basic, 8, basic, 0 if multiple < 1 else 1]),
        # 9
        VertexInfo(VertexType.Driven, [8, multiple * basic, 9, (0.0001 + abs(multiple - 1)) * basic, 1]),  # 10
    ]
    extra_lines = [
        [1, 2]
    ]
    driver = 2

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

    return Linkage(info, extra_lines, None, [5, 10], 2)


class LinkageBuilder:
    def __init__(self, global_color_hint: Tuple[float, float, float] = None):
        self.infos: List[VertexInfo] = []
        self.extra_lines: List[List[int]] = []
        self.colors: List[Tuple[float, float, float]] = []
        self.global_color_hint = global_color_hint if global_color_hint is not None else (0.28, 0.68, 0.99)
        self.tracked = []
        self.driver = -1

    def register_color(self, old_n: int, color_hint: Tuple[float, float, float]):
        color = color_hint if color_hint is not None else self.global_color_hint

        if len(self.colors) < self.vertices():
            self.colors.extend([color] * (self.vertices() - len(self.colors)))
        for i in range(old_n, self.vertices()):
            self.colors[i] = color

    # add Peaucellier straight line
    # need: nothing
    # return: id of x
    def add_straight_line(self, start: float = None, end: float = None,
                          color_hint: Tuple[float, float, float] = None) -> int:
        n: int = self.vertices()

        # 0.6799779794256628, 5.320021789745519

        mid = 3.0
        scale = 1
        if start is not None:
            mid = (start + end) / 2
            scale = (mid - start) / 2.32
            # print(mid, scale)
        # scale = 1
        self.infos.extend([
            VertexInfo(VertexType.Fixed, [mid, -7.5 * scale]),  # +0
            VertexInfo(VertexType.Fixed, [mid, -4.5 * scale]),  # +1
            VertexInfo(VertexType.Driver, [mid, -4.5 * scale, 3.0 * scale, math.pi / 2 - 0.6, math.pi / 2 + 0.6]),  # +2
            VertexInfo(VertexType.Driven, [n, 7.0 * scale, n + 2, 2.0 * scale, 0]),  # +3
            VertexInfo(VertexType.Driven, [n, 7.0 * scale, n + 2, 2.0 * scale, 1]),  # +4
            VertexInfo(VertexType.Driven, [n + 3, 2.0 * scale, n + 4, 2.0 * scale, 0]),  # +5, x-axis
        ])
        self.extra_lines.append([n + 1, n + 2])
        self.driver = n + 2

        self.register_color(n, color_hint)
        self.track([n + 5])
        return self.vertices() - 1

    # add origin
    # need: nothing
    # return: id of origin
    def add_origin(self, color_hint: Tuple[float, float, float] = None) -> int:
        self.infos.extend([VertexInfo(VertexType.Fixed, [0.0, 0.0])])  # +0

        self.register_color(self.vertices() - 1, color_hint)
        return self.vertices() - 1

    # add axes from straight line
    # need: id of origin and x
    # return: id of y
    def add_axes(self, o: int, x: int, color_hint: Tuple[float, float, float] = None) -> int:
        n: int = self.vertices()

        basic = 12.8

        self.infos.extend([
            VertexInfo(VertexType.Driven, [o, basic, x, basic, 1]),  # +0
            VertexInfo(VertexType.Driven, [o, basic, x, basic, 0]),  # +1
            VertexInfo(VertexType.Driven, [o, basic, n + 0, basic * math.sqrt(2), 0]),  # +2
            VertexInfo(VertexType.Driven, [o, basic, n + 1, basic * math.sqrt(2), 0]),  # +3
            VertexInfo(VertexType.Driven, [n + 2, basic, n + 3, basic, 1]),  # +4, y-axis
        ])

        self.register_color(n, color_hint)
        self.track([n + 4])
        return self.vertices() - 1

    # add zoomer (constant multiplier)
    # need: id of origin, zoomee, multiple
    # return: id of result
    def add_zoomer(self, o: int, x: int, multi: float = 2, color_hint: Tuple[float, float, float] = None) -> int:
        n: int = self.vertices()
        basic = 6.5

        shorter = (0.0001 + abs(multi - 1)) * basic

        self.infos.extend([
            VertexInfo(VertexType.Driven, [x, basic, o, basic, 1]),  # +0
            VertexInfo(VertexType.Driven, [o, multi * basic, n + 0, shorter, 1]),  # +1
            VertexInfo(VertexType.Driven, [x, shorter, n + 1, basic, 0 if multi < 1 else 1]),  # +2
            VertexInfo(VertexType.Driven, [n + 1, multi * basic, n + 2, shorter, 1]),  # +3
        ])

        self.register_color(n, color_hint)
        self.track([n + 3])
        return self.vertices() - 1

    # add vector adder
    # need: id of origin, op1, op2
    # return id of op1+op2
    def add_adder(self, o: int, a: int, b: int, color_hint: Tuple[float, float, float] = None) -> int:
        n: int = self.vertices()
        basic1 = 6
        basic2 = 6
        # print("n=", n, " o=", o, " a=", a, " b=", b)

        self.infos.extend([
            VertexInfo(VertexType.Driven, [o, basic2, a, basic1, 1]),  # +0
            VertexInfo(VertexType.Driven, [o, basic1, b, basic2, 1]),  # +1
            VertexInfo(VertexType.Driven, [n + 0, basic1 + 1e-4, n + 1, basic2 + 1e-4, 0, o]),  # +2
            VertexInfo(VertexType.Driven, [a, basic1, n + 2, basic2, 1, n + 0]),  # +3, not equal to +0
            VertexInfo(VertexType.Driven, [n + 2, basic1, b, basic2, 1, n + 1]),  # +4, not equal to +1
            VertexInfo(VertexType.Driven, [n + 3, basic1 + 1e-4, n + 4, basic2 + 1e-4, 1, n + 2]),  # +5
        ])

        self.register_color(n, color_hint)
        self.track([n + 5])
        return self.vertices() - 1

    # oa - ob = ba, and need move its start point to o, so we need to cal `bo + ba`.
    def add_suber(self, o: int, a: int, b: int, color_hint: Tuple[float, float, float] = None) -> int:
        n: int = self.vertices()
        self.add_adder(b, a, o)
        self.register_color(n, color_hint)
        return self.vertices() - 1

    # add a mover (constant adder)
    # need: id of x, (dx, dy)
    # return id of moved point
    # TODO: refine direction hint logic of +4
    def add_mover(self, x: int, dx: float, dy: float, color_hint: Tuple[float, float, float] = None) -> int:
        n: int = self.vertices()
        basic = 24.8

        x0 = random.uniform(-5, 0)
        y0 = random.uniform(-5, 0)
        # x0 = -3
        # y0 = 3

        d = math.sqrt(dx * dx + dy * dy)

        self.infos.extend([
            VertexInfo(VertexType.Fixed, [x0, y0]),  # +0
            VertexInfo(VertexType.Fixed, [x0 + dx, y0 + dy]),  # +1
            VertexInfo(VertexType.Driven, [x, basic, n + 0, basic, 0]),  # +2
            VertexInfo(VertexType.Driven, [n + 1, basic, n + 2, d, 0, n + 0]),  # +3
            VertexInfo(VertexType.Driven, [n + 3, basic, x, d, 1, n + 2]),  # +4
        ])
        self.add_extra_lines([[n + 0, n + 1]])

        self.register_color(n, color_hint)
        self.track([n + 4])
        return self.vertices() - 1

    # add an inverter
    # need: id of o and x
    # return: id of inverted index `t`
    # ot * ox = (a^2 - b^2)
    def add_inverter(self, o: int, x: int, color_hint: Tuple[float, float, float] = None) -> int:
        n: int = self.vertices()

        basic = 12.8
        tx = math.sqrt(basic * basic - 3)

        self.infos.extend([
            VertexInfo(VertexType.Driven, [o, basic, x, tx, 0]),  # n
            VertexInfo(VertexType.Driven, [o, basic, x, tx, 1]),  # n+1
            VertexInfo(VertexType.Driven, [n, tx, n + 1, tx, 0, x]),  # n+2
        ])
        # self.extra_lines.append([n + 1, n + 2])

        self.register_color(n, color_hint)
        self.track([n + 2])
        return self.vertices() - 1

    # add a squarer
    # need: id or o and x
    # return: id of squared point `t`
    # ot = ox^2
    # p^2 = 2/( 1/(p-1) - 1/(p+1) ) + 1
    def add_squarer(self, o: int, x: int, color_hint: Tuple[float, float, float] = None) -> int:
        n: int = self.vertices()

        psub1 = self.add_mover(x, -1, 0)
        padd1 = self.add_mover(x, 1, 0)
        #
        inv_sub = self.add_inverter(o, psub1, (1, 1, 0))
        inv_add = self.add_inverter(o, padd1, (1, 1, 0))
        subed = self.add_suber(o, inv_sub, inv_add)
        #
        inved = self.add_inverter(o, subed)
        inv2 = self.add_zoomer(o, inved, 2)
        #
        inv2 = self.add_mover(inv2, 1, 0)

        self.register_color(n, color_hint)
        return self.vertices() - 1

    # add fixed point
    # need: nothing
    # return: id of point
    def add_fixed(self, x: float = 0.0, y: float = 0.0, color_hint: Tuple[float, float, float] = None) -> int:
        self.infos.extend([VertexInfo(VertexType.Fixed, [x, y])])  # +0

        self.register_color(self.vertices() - 1, color_hint)
        return self.vertices() - 1

    def vertices(self):
        return len(self.infos)

    def get_linkage(self):
        return Linkage(self.infos, self.extra_lines, self.colors, self.tracked, self.driver)

    # set p as the traced point (config it's color), and return the linkage
    def set_color(self, p: int, color: Tuple[float, float, float]):
        # print("set_color", p, color)
        # print("self.colors", len(self.colors))
        self.colors[p] = color

    def add_extra_lines(self, lines: List[List[int]]):
        self.extra_lines.extend(lines)

    def track(self, points: List[int]):
        self.tracked.extend(points)


def YEqualKx(k: float = 2.0) -> Linkage:
    builder = LinkageBuilder()
    x = builder.add_straight_line()
    o = builder.add_origin()
    y = builder.add_axes(o, x)
    y2 = builder.add_zoomer(o, y, k)
    p = builder.add_adder(o, x, y2)

    # builder.set_color(p, (1.0, 0.0, 0.0))
    builder.track([p])
    builder.add_extra_lines([[o, x], [o, y], [o, p]])

    return builder.get_linkage()


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
    if (distCursor <= zone and notTrack != 0):
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
        if (n == driver):
            paint_point(pos=pos, size=0.8, cursor=cursor, zone=30., strength=.8, color=red, notTrack=1)
        if (tracked[n][0] != 0):
            paint_point(pos=pos, size=0.8, cursor=cursor, zone=30., strength=1., color=yellow, notTrack=0)
        else:
            paint_point(pos=pos, size=0.4, cursor=cursor, zone=30., strength=.6, color=purple, notTrack=1)


@ti.kernel
def paint_bg(color: ti.math.vec3, isPreview: ti.u8):
    for x, y in pixels:
        rgb = color
        if (isPreview != 0):
            # pixels[x, y] *= 0.9
            # pixels[x,y] -= pixels[x,y]/100
            pixels[x, y] -= 0.03 - pixels[x, y] * 0.01
        else:
            pixels[x, y] = rgb


@ti.func
def trans_pos(pos: ti.math.vec2) -> ti.math.vec2:
    position = (pos + (5, 10)) * 40
    return position


@ti.kernel
def ish_paint_line(vertices: ti.template(), indices: ti.template(), color: ti.math.vec3, cursor: ti.math.vec2):
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


def Squarer() -> Linkage:
    builder = LinkageBuilder()
    o = builder.add_fixed(color_hint=(1, 1, 1))
    x = builder.add_straight_line(1.5, 3)
    x2 = builder.add_squarer(o, x)
    y = builder.add_axes(o, x2)
    # builder.set_color(x, (0.0, 1.0, 0.0))
    #
    # x2 = builder.add_squarer(o, x)
    # builder.set_color(y, (0.0, 0.0, 1.0))
    p = builder.add_adder(o, x, y)

    builder.set_color(p, (0.0, 1.0, 0.0))
    builder.set_color(x, (1.0, 0.0, 0.0))
    builder.set_color(y, (1.0, 0.0, 0.0))
    builder.add_extra_lines([[p, x], [p, y]])

    return builder.get_linkage()


def YEqInvX() -> Linkage:
    builder = LinkageBuilder()
    o = builder.add_fixed()
    x = builder.add_straight_line(0.35, 6)
    inv = builder.add_inverter(o, x)
    y = builder.add_axes(o, inv)
    p = builder.add_adder(o, x, y)
    builder.set_color(p, (1.0, 0.0, 0.0))
    return builder.get_linkage()


def basic_adder() -> Linkage:
    builder = LinkageBuilder()
    o = builder.add_fixed(0, 0)
    a = builder.add_fixed(1, 0)
    b = builder.add_fixed(0.5, 0)
    # a = builder.add_fixed(8, 0)
    # b = builder.add_straight_line(color_hint=(0, 0, 0))
    # a = builder.add_zoomer(o, b, 1.5, (0, 0, 0))
    # p = builder.add_adder(b, a, o)
    p = builder.add_adder(b, a, o)
    builder.set_color(p, (1.0, 0.0, 0.0))
    return builder.get_linkage()


def Line() -> Linkage:
    builder = LinkageBuilder()
    x = builder.add_straight_line(1, 5)

    o = builder.add_fixed()
    return builder.get_linkage()


def main():
    dt = 0.01  # <= 0.01, 0.01 is slowest.
    substeps = int(1 / 100 // dt)
    isRun = 1
    isPreview = 0

    # # linkage = GrashofFourBarLinkage(3)
    # name = sys.argv[1] if len(sys.argv) > 1 else "Multiplier"
    # linkage = eval(name + "()")

    linkage = YEqualKx(0.5)

    # p = builder.add_inverter(o, x)
    # p = builder.add_squarer(o, x)

    # linkage = Squarer()
    # linkage = basic_adder()
    # result_dir = "/Users/lf/llaf/linkage-tc/results"
    # video_manager = ti.tools.VideoManager(output_dir=result_dir, framerate=24, automatic_build=False)

    window = ti.ui.Window("Leafall Linkage", (768, 768))
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(0, 0, 5)
    camera.lookat(0, 0, 0)

    current_t = 0.0
    steps = 0

    step_diff = 1

    trackedPoints = ti.Vector.field(2, dtype=ti.f32, shape=(linkage.get_trackedNum(), 120))

    while window.running:
        linkage.substep(steps)
        vertices = linkage.get_vertices()
        indices = linkage.get_indices()
        isTracked = linkage.get_istracked()
        cursor = ti.math.vec2(window.get_cursor_pos()) * res

        if (steps < 145):
            get_tracked_points(vertices, isTracked, trackedPoints, steps)

        # print(trackedPoints)

        # if isRun != 0:
        steps += step_diff
        current_t += dt

        paint_bg(black, isPreview)
        # print(linkage.get_vertices())

        if window.get_event(ti.ui.PRESS):
            if window.event.key == ' ':  # space
                step_diff = 1 - step_diff
            if window.event.key == 'Return':  # 1
                if (isPreview != 0):
                    isPreview = 0
                else:
                    isPreview = 1
                # isPreview = 1 if !isPreview else isPreview = 0

        driver = linkage.get_driver()

        create_points(vertices, cursor, isTracked, driver)

        # paint_track(vertices, isTracked)
        if (isPreview != 1):
            ish_paint_line(vertices, indices, purple, cursor)

        paint_track(steps, trackedPoints=trackedPoints, cursor=cursor)

        canvas.set_image(pixels)
        # video_manager.write_frame(window.get_image_buffer_as_numpy())

        window.show()
        # print("call_time = ", call_time[0])

        # sleep(200)
    # video_manager.rmake_video(gif=True, mp4=False)


if __name__ == '__main__':
    main()
