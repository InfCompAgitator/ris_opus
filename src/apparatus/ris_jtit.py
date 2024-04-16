from src.parameters import RIS_ACTIVE
from src.visibility_tools.visibility_graph_ctrl import VisibilityGraphCtrl
from src.main_controller import SimulationController
from src.visibility_tools.visibility_graph import Edge, Point, visible_vertices
from itertools import repeat, compress
from multiprocessing import Pool
from src.parameters import N_HOPS_ALLOWED, RELAYS_REQUIRED_RATE, RELAYS_TRANSMISSION_POWER, MAXIMUM_PL
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain as iter_chain
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from src.channel_model.mmwave_modeling import get_throughput_5g, get_throughput
from src.math_tools import db2lin
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib as mpl
import scipy.ndimage
from matplotlib import markers, lines, colors as mcolors
import os


mpl.rc('font', family='Times New Roman')

with_ris_color = 'r'
without_ris_color = 'blue'
with_ris_ls = '-'
without_ris_ls = '-.'

_markers = list(markers.MarkerStyle.markers.keys())
_markers.remove('none')
_markers.remove('None')
_markers.remove('')
_markers.remove(' ')
_markers.remove('1')
_markers.remove('2')
_markers.remove('3')
_markers.remove('4')
_markers.remove('8')
_markers.remove('.')
_markers.remove(',')
_colors = list(mcolors.BASE_COLORS.keys())
_colors[-1] = 'purple'

N_HOPS_MAX = 9


class Wall:
    def __init__(self, start_coords, end_coords, ris_feasible, orientation, wall_normal=None):
        self.start_coords = start_coords
        self.end_coords = end_coords
        self.ris_feasible = ris_feasible
        self.ris_installed = False
        self.orientation = orientation
        self.wall_normal = wall_normal

    def __repr__(self):
        return f"Wall at x: {self.start_coords[0], self.end_coords[0]}, y: {self.start_coords[1], self.end_coords[1]}"


class RIS:
    def __init__(self, x, y, wall, active_flag=RIS_ACTIVE):
        self.x = x
        self.y = y
        self.wall = wall
        self.active_flag = active_flag


class RisController(VisibilityGraphCtrl):
    shortest_paths = None
    maximum_path_loss = MAXIMUM_PL

    def __init__(self, sim_controller):
        super().__init__(sim_controller)
        self.ris_list = []
        self.walls = []
        self.populate_walls()
        self.segments, self.walls_normal = self.get_obstacles_segments()

    def get_obstacles_segments(self, flatten=True):
        segs, walls_normal = self.obstacles_objects.get_total_segments()
        return (segs if not flatten else [item for sublist in segs for item in sublist]), walls_normal

    def populate_walls(self, polygons=False):
        segs, walls_normal = self.get_obstacles_segments(False)

        # remove boundaries
        for i in range(4):
            segs.pop(0)

        [min_x, max_x], [min_y, max_y] = self.obstacles_objects.get_boundaries()
        margin_from_boundaries = 20
        min_x += margin_from_boundaries
        min_y += margin_from_boundaries
        max_x -= margin_from_boundaries
        max_y -= margin_from_boundaries
        if not polygons:
            for i in range(0, len(segs), 4):
                # min_bldg_x = min(segs[0][0], segs[1][0], segs[2][0])
                max_bldg_x = max(segs[i][0][0], segs[i][1][0], segs[i + 1][0][0], segs[i + 1][1][0])
                # min_bldg_y = min(segs[0][1], segs[1][1], segs[2][1])
                max_bldg_y = max(segs[i + 0][0][1], segs[i][1][1], segs[i + 1][0][1], segs[i + 1][1][1])
                for _seg in segs[i:i + 4]:
                    if _seg[0][0] != _seg[1][0]:
                        if _seg[0][1] == max_bldg_y:
                            orientation = 'u'
                        else:
                            orientation = 'd'
                    else:
                        if _seg[0][0] == max_bldg_x:
                            orientation = 'r'
                        else:
                            orientation = 'l'

                    if min_x < _seg[0][0] < max_x and min_y < _seg[0][1] < max_y \
                            and min_x < _seg[1][0] < max_x and min_y < _seg[1][1] < max_y:
                        self.walls.append(Wall(_seg[0], _seg[1], True, orientation))
                    else:
                        self.walls.append(Wall(_seg[0], _seg[1], False, orientation))
        else:
            for _seg, wall_normal in zip(segs, walls_normal):
                if min_x < _seg[0][0] < max_x and min_y < _seg[0][1] < max_y \
                        and min_x < _seg[1][0] < max_x and min_y < _seg[1][1] < max_y:
                    self.walls.append(Wall(_seg[0], _seg[1], True, None, wall_normal))
                else:
                    self.walls.append(Wall(_seg[0], _seg[1], False, None, wall_normal))

    def add_edges_to_visibility_graph(self, point_1, point_2, ris):
        _edge = Edge(point_1, point_2)
        _edge.ris_hop = ris
        if point_2 in self.vis_graph.visgraph.get_adjacent_points(point_1):
            old_edge = next(
                x for x in ris_ctrl.vis_graph.visgraph.graph[point_1] if x == _edge)
            if old_edge.get_edge_cost(True) < _edge.get_edge_cost(True):
                return
            else:
                self.vis_graph.visgraph.edges.remove(old_edge)
                self.vis_graph.visgraph.graph[old_edge.p1].remove(old_edge)
                self.vis_graph.visgraph.graph[old_edge.p2].remove(old_edge)
        self.vis_graph.visgraph.add_edge(_edge)

    def install_ris_to_wall(self, idx):
        wall = self.walls[idx]
        if wall.ris_installed:
            print("RIS already installed here")
            return
        wall.ris_installed = True

        if wall.wall_normal is not None:
            x_scale = -1*abs(wall.wall_normal[0]) if wall.start_coords[0] > wall.end_coords[0] else abs(wall.wall_normal[0])
            y_scale = -1*abs(wall.wall_normal[1]) if wall.start_coords[1] > wall.end_coords[1] else abs(wall.wall_normal[1])
            ris_xs = np.array([wall.start_coords[0] + x_scale, wall.end_coords[0] - x_scale])
            ris_ys = np.array([wall.start_coords[1] + y_scale, wall.end_coords[1] - y_scale])
        else:
            scale = 10
            ris_xs = np.array([wall.start_coords[0] + scale, wall.end_coords[0] - scale])
            ris_ys = np.array([wall.start_coords[1] + scale, wall.end_coords[1] - scale])

        if wall.wall_normal is None:
            # # remove for harder
            shift = 0.01
            # n_idx = idx / 4
            # number_dec = n_idx - int(n_idx)
            # n_idx = number_dec * 4

            if wall.orientation == 'l':
                ris_xs -= shift
            elif wall.orientation == 'd':
                ris_ys = ris_ys - shift
            elif wall.orientation == 'r':
                ris_xs += shift
            else:
                ris_ys = ris_ys + shift
        else:
            ris_xs += wall.wall_normal[0]
            ris_ys += wall.wall_normal[1]

        ris_obj = RIS(ris_xs, ris_ys, wall)
        self.ris_list.append(ris_obj)
        return ris_obj

    def get_ris_visibility(self, ris_obj):
        x = ris_obj.x.sum() / 2
        y = ris_obj.y.sum() / 2
        return visible_vertices(Point(x,y), self.vis_graph.graph)

    def install_ris_and_update_visibility(self, wall_idx):
        if not self.walls[wall_idx].ris_feasible or self.walls[wall_idx].ris_installed:
            print("RIS already installed at", self.walls[wall_idx])
            return False
        new_ris = self.install_ris_to_wall(wall_idx)
        vertices_within_ris = self.get_ris_visibility(new_ris)
        vertices_within_ris = list(set(vertices_within_ris))
        self.make_vertices_visible(vertices_within_ris, new_ris)
        return True

    def make_vertices_visible(self, vertices_within_ris, ris):
        for i, pt_1 in enumerate(vertices_within_ris):
            for j, pt_2 in enumerate(vertices_within_ris):
                if i == j:
                    continue
                self.add_edges_to_visibility_graph(pt_1, pt_2, ris)

    def get_shortest_path_to_dest(self, point_dest):
        if self.src is None:
            raise Exception("No source set!")
        return self.vis_graph.shortest_path \
            (Point(self.src[0], self.src[1]), point_dest, [], maximum_path_loss=self.maximum_path_loss)

    def get_shortest_paths_to_vertices(self):
        # remove for harder
        pool = Pool(4)
        results = list(pool.imap(self.get_shortest_path_to_dest, self.total_vertices, chunksize=10))
        self.shortest_paths = results
        # self.shortest_paths = [self.get_shortest_path_to_dest(_vert) for _vert in self.total_vertices]

        return self.shortest_paths

    def get_achievable_throughput_to_vertices(self):
        vertices_throughput = np.zeros(self.n_vertices, dtype=float)
        vertices_throughput[0] = np.inf
        for vert_idx, (_path, _) in enumerate(self.shortest_paths):
            if vert_idx == 0:
                continue
            min_througput = np.inf
            for hop_idx, _hop in enumerate(_path):
                if not hop_idx:
                    continue
                # remove for harder
                try:
                    res_idx = self.total_vertices[:vert_idx].index(_hop)
                    hop_throughput = vertices_throughput[res_idx]
                except:
                    _edge = next(
                        x for x in ris_ctrl.vis_graph.visgraph.graph[_hop] if x == Edge(_hop, _path[hop_idx - 1]))
                    # if _edge.ris_hop:
                    #     if _edge.ris_hop.x == 1007.0 or _edge.ris_hop.x == 207:
                    #         raise Exception("Found it!", _edge.ris_hop.x)
                    path_loss = _edge.get_edge_cost(path_loss=True)
                    hop_throughput = get_throughput_5g(RELAYS_TRANSMISSION_POWER / db2lin(path_loss))
                min_througput = min(min_througput, hop_throughput)
            vertices_throughput[vert_idx] = min_througput
        return vertices_throughput

    def get_n_hops_to_dest(self, dest):
        return len(self.get_shortest_path_to_dest(dest))

    def get_pts_reachable_with_n_hops(self, n_hops=N_HOPS_ALLOWED):
        flags = [flag and len(i) < n_hops + 2 for i, flag in self.shortest_paths]
        return list(compress(self.total_vertices[1:], flags[1:]))

    def random_ris_installation(self, n_walls=20):
        random_idxs = np.random.permutation(len(self.walls))
        ctr, idx = 0, 0
        while (ctr < n_walls):
            if (self.install_ris_and_update_visibility(random_idxs[idx])):
                ctr += 1
            idx += 1

    def get_square_walls_idxs(self, three_squares):
        selected_walls = []
        # xs = [147 + i * 400 for i in range(3)]
        # xs += [267 + i * 400 for i in range(3)]
        # ys = [267 + i * 570 for i in range(3)]
        # ys += [423 + i * 570 for i in range(3)]
        # xs = [147 + 400]
        # xs += [267 + 400]
        xs = []
        if three_squares:
            xs += [147]
            xs += [267]
            xs += [147 + 800]
            xs += [267 + 800]
        xs += [147 + 400]
        xs += [267 + 400]
        ys = [267 + 570]
        ys += [423 + 570]

        for idx, _wall in enumerate(self.walls):
            if _wall.start_coords[0] in xs and _wall.end_coords[0] in xs and \
                    _wall.start_coords[1] in ys and _wall.end_coords[1] in ys:
                selected_walls.append(idx)
        return selected_walls

    def install_ris_square(self, three_squares=False):
        idxs = self.get_square_walls_idxs(three_squares)
        for _idx in idxs:
            self.install_ris_and_update_visibility(_idx)

    def plot_ris(self, ax):
        if not self.ris_list:
            return
        for idx, _ris in enumerate(self.ris_list):
            xs, ys = zip(_ris.wall.start_coords, _ris.wall.end_coords)
            if not idx:
                ax.plot(xs, ys, c='black', label='RIS', linestyle='solid', linewidth=4)
            else:
                ax.plot(xs, ys, c='black', linestyle='solid', linewidth=4)

    def plot_points(self, ax, points, ris_involved=False):
        for idx, _vertex in enumerate(points):
            if not idx:
                ax.plot(_vertex.x, _vertex.y, c=('green' if not ris_involved else 'red'), marker='o',
                        label=f'Max {N_HOPS_ALLOWED} hops' + \
                              (' and RIS' if ris_involved else ''),
                        linestyle='none')
            else:
                ax.plot(_vertex.x, _vertex.y, c=('green' if not ris_involved else 'red'), marker='o')

    def plot_results(self, points):
        fig, ax = plt.subplots
        self.plot_ris(ax)
        self.plot_points(ax, points)
        self.plot_obstacles(ax)

        handles, labels = ax.get_legend_handles_labels()

        def flip(items, ncol):
            return iter_chain(*[items[i::ncol] for i in range(ncol)])

        plt.legend(flip(handles, 2), flip(labels, 2), loc='upper left', ncol=2)
        plt.show()

    def plot_throughput_heatmap(self, rates, rates_ris, rates_ris_all_squares, min_val, max_val):
        fig = plt.figure(figsize=(9.75 * (1.5 if rates_ris_all_squares is not None else 1), 4))
        grid = ImageGrid(fig, 111,  # as in plt.subplot(111)
                         nrows_ncols=(1, (3 if rates_ris_all_squares is not None else 2)),
                         axes_pad=0.15,
                         share_all=True,
                         cbar_location="right",
                         cbar_mode="single",
                         cbar_size="7%",
                         cbar_pad=0.15,
                         aspect=False
                         )
        cmap = plt.get_cmap('viridis')
        bounds = np.arange(min_val // 20e6 * 20 - 20, 260, 20)
        # norm = colors.BoundaryNorm(bounds, cmap.N)
        xs = np.array([_vert.x for _vert in self.total_vertices])
        ys = np.array([_vert.y for _vert in self.total_vertices])
        inf_idxs = np.where(rates == np.inf)
        rates[inf_idxs] = 1
        rates[inf_idxs] = rates.max()
        grid[0].tricontourf(xs, ys, rates / 1e6, cmap=cmap, levels=bounds)
        inf_idxs = np.where(rates_ris == np.inf)
        rates_ris[inf_idxs] = 1
        rates_ris[inf_idxs] = rates_ris.max()
        im = grid[1].tricontourf(xs, ys, rates_ris / 1e6, cmap=cmap, levels=bounds)
        if rates_ris_all_squares is not None:
            inf_idxs = np.where(rates_ris_all_squares == np.inf)
            rates_ris_all_squares[inf_idxs] = 1
            rates_ris_all_squares[inf_idxs] = rates_ris_all_squares.max()
            im = grid[2].tricontourf(xs, ys, rates_ris_all_squares / 1e6, cmap=cmap, levels=bounds)
            grid[2].set_xlabel('Two RISs')
        grid[0].set_xlabel('No RIS')
        grid[1].set_xlabel('Single RIS')
        cbar = fig.colorbar(im, cax=grid[1].cax, ticks=bounds, spacing='uniform')
        cbar.set_ticks(bounds)  # TODO: Removed true as second parameter
        cbar.set_label("Throughput [Mbps]")
        # # Colorbar
        #
        # cb = grid[1].cax.colorbar(im)
        # cb.set_label("Throughput [Mbps]")
        # grid[1].cax.toggle_label(True)
        fig.show()

    def plot_rates_cdf(self, rates, rates_ris):
        fig, ax = plt.subplots()

        def ecdf_x_y(a):
            def ecdf(a):
                x, counts = np.unique(a, return_counts=True)
                cusum = np.cumsum(counts)
                return x, cusum / cusum[-1]

            x, y = ecdf(a)
            x = np.insert(x, 0, x[0])
            y = np.insert(y, 0, 0.)
            return x, y

        x, y = ecdf_x_y(rates)
        ax.plot(x, y, drawstyle='steps-post', label='RIS', color=without_ris_color,
                ls=with_ris_ls, alpha=0.7)
        ax.grid(True)
        fig.show()

    def plot_n_reachable_pts_per_n_hops(self, n_pts_per_n_hops_ris, n_pts_per_n_hops):
        def autolabel(rects):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 1),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')  # , rotation=45)

        fig, ax = plt.subplots()
        X = [f'{i + 1} hops' for i in range(len(n_pts_per_n_hops_ris))]
        width = 1
        X_axis = np.arange(width, width * (len(X) + 1), width)
        # shift = (len(num_users) * 3 * width) + 2 * width
        shift = 0.22
        # X_axis = np.array([np.arange(width, width * (len(X)), width) + i * shift for i in range(len(num_users))]).flatten()
        rects1 = ax.bar(X_axis + shift * width, n_pts_per_n_hops_ris, label='RIS', width=0.3 * width,
                        color=with_ris_color,
                        align='center')
        rects2 = ax.bar(X_axis - shift * width, n_pts_per_n_hops, label='No RIS', width=0.3 * width,
                        color=without_ris_color,
                        align='center')
        ax.set_xticks(range(1, len(n_pts_per_n_hops_ris) + 1))
        ax.set_xlabel('N')
        ax.set_ylabel('Number of reachable hops')
        plt.legend(loc='upper left', ncol=1)
        autolabel(rects1)
        autolabel(rects2)
        fig.show()
    def save_rechable_pts_for_ns(self, folder_name=None):
        prev_pts = set()
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        for i in range(1, N_HOPS_MAX):
            points = set(self.get_pts_reachable_with_n_hops(i)) ^ prev_pts
            _pts_tuples = [(_p.x, _p.y) for _p in points]
            np.save(folder_name+f'n_hops_equal_{i}.npy', np.array(_pts_tuples))
    def plot_reachable_pts_for_ns_all(self):
        fig, ax = plt.subplots()
        self.plot_obstacles(ax)
        self.plot_endpoints(ax)
        self.plot_ris(ax)
        prev_pts = set()
        for i in range(1, N_HOPS_MAX):
            points = set(self.get_pts_reachable_with_n_hops(i)) ^ prev_pts
            prev_pts = prev_pts.union(points)
            for idx, _vertex in enumerate(points):
                if not idx:
                    ax.plot(_vertex.x, _vertex.y, c=_colors[(i - 2) % len(_colors)], marker=_markers[i % len(_markers)],
                            label=f'{i} hops',
                            linestyle='none')
                else:
                    ax.plot(_vertex.x, _vertex.y, c=_colors[(i - 2) % len(_colors)], marker=_markers[i % len(_markers)])
        ax.legend(loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.15))
        fig.subplots_adjust(left=0.12, top=0.85, right=0.95, bottom=0.15, wspace=0.1, hspace=0.1)
        fig.show()

    def install_walls_poznan(self, idx):
        idxs = [163, 164, 177, 178, 408, 409, 455, 456]
        self.install_ris_and_update_visibility(idxs[idx])


if __name__ == '__main__':
    results_folder_no_ris = os.path.join(os.getcwd(), "data_save_jtit\\no_ris\\")
    results_folder_single_ris = os.path.join(os.getcwd(), "data_save_jtit\\single_ris\\")
    results_folder_double_ris = os.path.join(os.getcwd(), "data_save_jtit\\double_ris\\")
    sim_ctrl = SimulationController()
    ris_ctrl = RisController(sim_ctrl)
    ris_ctrl.set_source_and_dest([600, 900], None)
    # ris_ctrl.install_walls_poznan(7)
    ris_ctrl.build_graph()
    # ris_ctrl.install_walls_poznan(0)
    #
    # # No RIS
    ris_ctrl.get_shortest_paths_to_vertices()
    reachable_pts = ris_ctrl.get_pts_reachable_with_n_hops()
    ris_ctrl.save_rechable_pts_for_ns(results_folder_no_ris)
    ris_ctrl.plot_reachable_pts_for_ns_all()
    rates = ris_ctrl.get_achievable_throughput_to_vertices()
    n_pts_per_n_hops = []
    for i in range(1, N_HOPS_MAX):
        n_pts_per_n_hops.append(len(ris_ctrl.get_pts_reachable_with_n_hops(i)))


    fig, ax = plt.subplots()
    ris_ctrl.plot_points(ax, reachable_pts)
    ris_ctrl.plot_obstacles(ax)
    ris_ctrl.plot_endpoints(ax)
    #
    # # Single RIS RIS
    ris_ctrl.install_ris_square()
    # ris_ctrl.install_walls_poznan(3)
    ris_ctrl.get_shortest_paths_to_vertices()
    ris_ctrl.plot_reachable_pts_for_ns_all()
    ris_ctrl.save_rechable_pts_for_ns(results_folder_single_ris)
    reachable_pts_ris = ris_ctrl.get_pts_reachable_with_n_hops()
    rates_ris = ris_ctrl.get_achievable_throughput_to_vertices()
    n_pts_per_n_hops_ris = []
    for i in range(1, N_HOPS_MAX):
        n_pts_per_n_hops_ris.append(len(ris_ctrl.get_pts_reachable_with_n_hops(i)))

    reachable_pts_new = list((set([]) | set(reachable_pts_ris)) ^ set(reachable_pts))
    ris_ctrl.plot_points(ax, reachable_pts_new, True)
    ris_ctrl.plot_ris(ax)
    ris_ctrl.show_legend(ax)
    ris_ctrl.plot_throughput_heatmap(rates, rates_ris, None, min(rates.min(), rates_ris.min()),
                                     max(rates.max(), rates_ris.max()))
    fig.show()

    # # 3 RIS RIS
    # ris_ctrl.install_ris_square(True)
    # # ris_ctrl.install_walls_poznan(7)
    # ris_ctrl.get_shortest_paths_to_vertices()
    # ris_ctrl.plot_reachable_pts_for_ns_all()
    # ris_ctrl.save_rechable_pts_for_ns(results_folder_double_ris)
    # reachable_pts_3_ris = ris_ctrl.get_pts_reachable_with_n_hops()
    # rates_ris_3 = ris_ctrl.get_achievable_throughput_to_vertices()
    # n_pts_per_n_hops_ris_3 = []
    # for i in range(1, N_HOPS_MAX):
    #     n_pts_per_n_hops_ris_3.append(len(ris_ctrl.get_pts_reachable_with_n_hops(i)))
    #
    # reachable_pts_new = list((set(reachable_pts_3_ris) | set(reachable_pts_ris)) ^ set(reachable_pts))
    #
    # ris_ctrl.plot_points(ax, reachable_pts_new, True)
    # ris_ctrl.plot_ris(ax)
    # ris_ctrl.show_legend(ax)
    # ris_ctrl.plot_throughput_heatmap(rates, rates_ris, rates_ris_3, min(rates.min(), rates_ris.min()),
    #                                  max(rates.max(), rates_ris.max()))
    # fig.show()

    # ##for barplots
    # sim_ctrl = SimulationController()
    # ris_ctrl = RisController(sim_ctrl)
    # ris_ctrl.set_source_and_dest([600, 900], None)
    # ris_ctrl.build_graph()
    # n_pts_per_n_hops = []
    # for snr_requirement in [85, 90]:
    #     ris_ctrl.maximum_path_loss = snr_requirement
    #     n_pts_per_n_hops_array = np.zeros(8)
    #     # No RIS
    #     ris_ctrl.get_shortest_paths_to_vertices()
    #     for i in range(1, 9):
    #         n_pts_per_n_hops_array[i-1] = len(ris_ctrl.get_pts_reachable_with_n_hops(i))
    #         if n_pts_per_n_hops_array[i-1] == len(ris_ctrl.total_vertices) - 1:
    #             break
    #     n_pts_per_n_hops.append(n_pts_per_n_hops_array[0:i])
    #
    # ris_ctrl.install_ris_square()
    # n_pts_per_n_hops_ris = []
    # for idx, snr_requirement in enumerate([85, 90]):
    #     ris_ctrl.maximum_path_loss = snr_requirement
    #     n_pts_per_n_hops_ris_array = np.zeros(8)
    #     # RIS
    #     ris_ctrl.get_shortest_paths_to_vertices()
    #     for i in range(1, 9):
    #         n_pts_per_n_hops_ris_array[i-1] = len(ris_ctrl.get_pts_reachable_with_n_hops(i))
    #     n_pts_per_n_hops_ris.append(n_pts_per_n_hops_ris_array[0:len(n_pts_per_n_hops[idx])])
    #
    # for i in range(len(n_pts_per_n_hops_ris)):
    #     ris_ctrl.plot_n_reachable_pts_per_n_hops(n_pts_per_n_hops_ris[i], n_pts_per_n_hops[i])
# (rates_ris_3[np.where(rates_ris_3 != rates_ris)] - rates_ris[np.where(rates_ris_3 != rates_ris)])/1e6

# reachable_pts_ris = ris_ctrl.get_pts_reachable_with_n_hops(5)
# reachable_pts = ris_ctrl.get_pts_reachable_with_n_hops(5)
# fig, ax = plt.subplots()
# ris_ctrl.plot_po)ints(ax, reachable_pts)
# ris_ctrl.plot_obstacles(ax)
# ris_ctrl.plot_endpoints(ax)
# ris_ctrl.plot_points(ax, reachable_pts_new, True)
# ris_ctrl.plot_ris(ax)
# ris_ctrl.show_legend(ax)
# fig.show()

# minx, maxx, miny, maxy = 580, 710, 1050, 1200
# for wall_idx, _wall in enumerate(ris_ctrl.walls):
#     if minx <= _wall.start_coords[0] <= maxx and minx <= _wall.end_coords[0] <= maxx and \
#             miny <= _wall.start_coords[1] <= maxy and miny <= _wall.end_coords[1] <= maxy:
#         print(wall_idx, _wall)

#PARAMETERS.py
# RIS_ACTIVE = True
# N_HOPS_ALLOWED = 8
# RELAYS_BANDWIDTH = 20e6
# RELAYS_BANDWIDTH_5G_NOISE = 18720000  # 38.306 model by Stroke
# RELAYS_BANDWIDTH_5G = 17472000  # 38.306 model by Stroke
# NOISE_SPECTRAL_DENSITY_RELAY = -174
# RELAYS_NOISE_POWER = db2lin(NOISE_SPECTRAL_DENSITY_RELAY - 30 + lin2db(RELAYS_BANDWIDTH))  # dBm input -> linear in W
# RELAYS_NOISE_POWER_5G = db2lin(
#     NOISE_SPECTRAL_DENSITY_RELAY - 30 + lin2db(RELAYS_BANDWIDTH_5G_NOISE))  # dBm input -> linear in W
# EFFICIENCY_5G_RATE_FACTOR = 0.82
# RELAYS_TRANSMISSION_POWER = 100e-3
# MAXIMUM_PL = 90  # 85, 100, 111 for 200, 100, 50 mbps
# ACTIVE_RIS_GAIN = 20  # dB
# RELAYS_REQUIRED_RATE = (200e6)
# USE_POZNAN = False