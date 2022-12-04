import math

from .builder import LinkageBuilder
from .linkage import Linkage, VertexInfo, VertexType


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
