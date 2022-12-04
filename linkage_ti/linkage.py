import enum
import math
from typing import List, Tuple

import taichi as ti

from .utils import intersect_of_circle


@enum.unique
class VertexType(enum.Enum):
    Fixed = 0
    Driver = 1
    Driven = 2


class VertexInfo:
    """vertex information class for linkage_ti system"""

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

        print("N =", self.N)

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
