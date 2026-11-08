import numpy as np

MAX_STEP_SQUARED = 8  # two pixels diagonally


def calculate(border, num_samples=500):
    edge_y, edge_x = np.where(border)
    points = np.column_stack((edge_x, edge_y))
    ordered_pts = [points[0]]
    unvisited = list(points[1:])
    while unvisited:
        dists = np.sum((unvisited - ordered_pts[-1])**2, axis=1)
        nearest = np.argmin(dists)
        if dists[nearest] > MAX_STEP_SQUARED:
            break
        ordered_pts.append(unvisited.pop(nearest))
    ordered_pts = np.array(ordered_pts)

    distances = np.sqrt(np.sum(np.diff(ordered_pts, axis=0)**2, axis=1))
    cumulative_dist = np.insert(np.cumsum(distances), 0, 0)
    target_dists = np.linspace(0, cumulative_dist[-1], num_samples, endpoint=False)

    x = np.interp(target_dists, cumulative_dist, ordered_pts[:, 0])
    y = np.interp(target_dists, cumulative_dist, ordered_pts[:, 1])
    return x, y
