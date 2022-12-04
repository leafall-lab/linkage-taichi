import math
from typing import List, Tuple

from .linkage import Linkage, VertexInfo, VertexType


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

    # set p as the traced point (config it's color), and return the linkage_ti
    def set_color(self, p: int, color: Tuple[float, float, float]):
        # print("set_color", p, color)
        # print("self.colors", len(self.colors))
        self.colors[p] = color

    def add_extra_lines(self, lines: List[List[int]]):
        self.extra_lines.extend(lines)

    def track(self, points: List[int]):
        self.tracked.extend(points)
