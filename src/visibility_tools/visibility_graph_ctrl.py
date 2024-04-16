import numpy as np

from src.main_controller import SimulationController
from src.visibility_tools.visibility_graph import Point, VisGraph
import matplotlib.pyplot as plt
import matplotlib;

# matplotlib.use("module://backend_interagg")
from src.environment.obstacles import ObstaclesPlotter
from itertools import chain as iter_chain


class VisibilityGraphCtrl(ObstaclesPlotter):
    obstacles_polys = None
    src, dest = None, None
    total_vertices = []
    selected_vertices = []
    n_vertices = 0
    vis_graph = None
    shortest_path = []

    def __init__(self, simulation_controller: SimulationController, build_graph=False):
        self._controller = simulation_controller
        self.obstacles_objects = self._controller.user_model.obstacles_objects
        self.populate_obstacles_polygons()
        # self.set_source_and_dest((0, 0), (500, 500))
        self.populate_selected_vertices()
        if build_graph:
            self.build_graph()

    def populate_obstacles_polygons(self):
        polys = []
        for _obstacle in self._controller.user_model.get_obstacles():
            poly = []
            for _vertex in _obstacle.vertices:
                poly.append(Point(_vertex[0], _vertex[1]))
            polys.append(poly)
        self.obstacles_polys = polys

    def set_source_and_dest(self, src=None, dest=None):
        if src is not None:
            self.src = src
        if dest is not None:
            self.dest = dest
        self.populate_selected_vertices()

    def populate_selected_vertices(self, extra_vertices=None):
        """Selected are those that are extra on top of corners"""
        self.selected_vertices = []
        if self.dest:
            self.selected_vertices.append(Point(self.dest[0], self.dest[1]))
        if self.src:
            self.selected_vertices.append(Point(self.src[0], self.src[1]))
        if extra_vertices:
            for _vert in extra_vertices:
                self.selected_vertices.append(Point(_vert[0], _vert[1]))
        self.update_total_vertices()

    def update_total_vertices(self, include_obstacles_corners=True):
        self.total_vertices = [self.selected_vertices] + (self.obstacles_polys if include_obstacles_corners else [])

        self.total_vertices = [item for sublist in self.total_vertices for item in sublist]
        self.total_vertices = list(set(self.total_vertices))
        self.n_vertices = len(self.total_vertices)

    def build_graph(self):
        self.vis_graph = VisGraph()
        input_polys = self.obstacles_polys + [[_vertex] for _vertex in self.selected_vertices]
        self.vis_graph.build(input_polys)

    def get_shortest_path(self):
        if not self.dest or not self.src:
            raise Exception("No source or destination were set!")
        self.shortest_path, _ = self.vis_graph.shortest_path \
            (Point(self.src[0], self.src[1]), Point(self.dest[0], self.dest[1]), [])

        self.shortest_path = self.shortest_path[1:-1]
        return self.shortest_path

    def plot_default_vertices(self, ax):
        for idx, _vertex in enumerate(self.total_vertices):
            if idx == 0:
                ax.plot(_vertex.x, _vertex.y, c='navy', linestyle='', marker='.', markersize=10, label='Possible Deployment Location')
            else:
                ax.plot(_vertex.x, _vertex.y, c='navy', marker='.', markersize=10)

    def plot_shortest_path(self, ax):
        if not self.shortest_path:
            return
        for _hop in self.shortest_path:
            ax.plot(_hop.x, _hop.y, c='red', marker='8', markeredgecolor='black', markersize=5)
        ax.plot([], [], linestyle='none', c='red', marker='8', markeredgecolor='black', markersize=5, label='DRS')
        path = [Point(self.src[0], self.src[1])] + self.shortest_path + [Point(self.dest[0], self.dest[1])]
        for idx in range(len(path) - 1):
            xs = [path[idx].x, path[idx + 1].x]
            ys = [path[idx].y, path[idx + 1].y]
            if not idx:
                ax.plot(xs, ys, c='red', label='mmWave link')
            else:
                ax.plot(xs, ys, c='red')

    def plot_endpoints(self, ax):
        # plt.scatter(self.src[0], self.src[1], c='blue', marker='s')
        # plt.scatter(self.dest[0], self.dest[1], c='blue', marker='s')
        if self.src:
            ax.plot(self.src[0], self.src[1], c='blue', marker='s', linestyle='none', label='MBS')
        if self.dest:
            ax.plot(self.dest[0], self.dest[1], c='green', marker='o', linestyle='none', label='Hotspot')

    def plot_results(self, show_flag=True):
        fig, ax = plt.subplots()
        self.plot_obstacles(ax)
        # self.plot_default_vertices(ax)
        self.plot_shortest_path(ax)
        self.plot_endpoints(ax)
        self.show_legend(ax)
        if show_flag:
            fig.show()
        return ax


if __name__ == '__main__':
    sim_ctrl = SimulationController()
    vis_graph = VisibilityGraphCtrl(sim_ctrl, False)
    # vis_graph.set_source_and_dest([645, 1125], [900, 950])
    vis_graph.set_source_and_dest([645, 1125], [870, 600])
    vis_graph.build_graph()
    vis_graph.get_shortest_path()
    vis_graph.plot_results()
    # fig, ax = plt.subplots()
    # vis_graph.plot_obstacles(ax)
    # vis_graph.plot_default_vertices(ax)
    # vis_graph.show_legend(ax)
    # fig.show()
