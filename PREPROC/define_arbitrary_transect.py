import numpy as np

'''
defines a control surface along an arbitrary transect, from the sea surface to the seabed.
The transect is a polyline through user defined (lon, lat) points: A, B, C, ...
Imported by create_fluxes.py (see compute_arbitrary_transect there).

Grid conventions (NEMO C-grid, same as define_isobath.py and plot_control_surfaces.py):
  - i increases eastward, j increases northward
  - F[j,i] is the north-east corner of T[j,i]
  - U[j,i] is the east face of T[j,i],  it runs between F[j-1,i] and F[j,i]
  - V[j,i] is the north face of T[j,i], it runs between F[j,i-1] and F[j,i]
  - in the model output the fluxes of the face (U|V, j, i) are saved at the T point (j,i),
    positive eastward (U faces) and northward (V faces)

Algorithm:
  1. each point is snapped to the nearest F node.
  2. consecutive nodes are connected by a path on the F lattice made of single U / V faces. 
     Among all the monotone staircases (all have the same, minimum, Manhattan length) 
     the one that stays closest to the straight line between the two nodes is chosen
     (dynamic programming, minimum sum of squared distances of the nodes from the line).
  3. each face is given a sign multiplier according to the direction in which the path crosses it,
     so that fluxes are positive from the left to the right of the transect,
     looking from point i to point i+1:
         step north (dy=+1): U face, multiplier +1 (the right is east)
         step south (dy=-1): U face, multiplier -1
         step east  (dx=+1): V face, multiplier -1 (the right is south)
         step west  (dx=-1): V face, multiplier +1
     W (vertical flux) is 0: the transect is a vertical wall.
  4. the faces are converted to bounmask indexes of the water points (all levels).
     The transect is continuous, but it can cross land: faces owned by a land cell have
     no index and are not recorded. 
'''

EARTH_RADIUS_KM = 6371.0


def to_km(lon, lat, lat_ref):
    # local equirectangular projection, to measure euclidean distances on a non-square grid
    x = np.radians(lon) * np.cos(np.radians(lat_ref)) * EARTH_RADIUS_KM
    y = np.radians(lat) * EARTH_RADIUS_KM
    return x, y


def snap_to_node(lon, lat, glamf, gphif):
    '''
    nearest F node (j, i) to a point
    '''
    if not (glamf.min() <= lon <= glamf.max() and gphif.min() <= lat <= gphif.max()):
        raise ValueError(f"point (lon={lon}, lat={lat}) is outside the domain "
                         f"(lon {glamf.min():.2f} - {glamf.max():.2f}, lat {gphif.min():.2f} - {gphif.max():.2f})")
    x, y = to_km(glamf, gphif, lat)
    x0, y0 = to_km(lon, lat, lat)
    j, i = np.unravel_index(np.argmin((x - x0) ** 2 + (y - y0) ** 2), glamf.shape)
    return int(j), int(i)


def staircase(node0, node1, glamf, gphif):
    '''
    path of F nodes from node0 to node1 with minimum Manhattan length, 
    that stays closest to the straight line between them.
    Returns the list of nodes (j, i), including both ends.
    '''
    j0, i0 = node0
    j1, i1 = node1
    di, dj = int(np.sign(i1 - i0)), int(np.sign(j1 - j0))
    ni, nj = abs(i1 - i0), abs(j1 - j0)
    lat_ref = 0.5 * (gphif[j0, i0] + gphif[j1, i1])
    xa, ya = to_km(glamf[j0, i0], gphif[j0, i0], lat_ref)
    xb, yb = to_km(glamf[j1, i1], gphif[j1, i1], lat_ref)
    dx, dy = xb - xa, yb - ya
    len2 = dx * dx + dy * dy

    def node_cost(a, b):
        # squared distance from the segment of the node reached after a steps in i and b steps in j
        j, i = j0 + dj * b, i0 + di * a
        x, y = to_km(glamf[j, i], gphif[j, i], lat_ref)
        t = np.clip(((x - xa) * dx + (y - ya) * dy) / len2, 0.0, 1.0)
        return (x - xa - t * dx) ** 2 + (y - ya - t * dy) ** 2

    cost = np.full((ni + 1, nj + 1), np.inf)
    from_i = np.zeros((ni + 1, nj + 1), dtype=bool)  # True: the last step was in i
    for a in range(ni + 1):
        for b in range(nj + 1):
            if a == 0 and b == 0:
                cost[a, b] = node_cost(a, b)
                continue
            c_i = cost[a - 1, b] if a > 0 else np.inf
            c_j = cost[a, b - 1] if b > 0 else np.inf
            from_i[a, b] = c_i <= c_j
            cost[a, b] = node_cost(a, b) + min(c_i, c_j)
    # backtrack
    a, b = ni, nj
    nodes = [(j0 + dj * b, i0 + di * a)]
    while (a, b) != (0, 0):
        if from_i[a, b]:
            a -= 1
        else:
            b -= 1
        nodes.append((j0 + dj * b, i0 + di * a))
    return nodes[::-1]


def nodes_to_edges(nodes):
    '''
    faces crossed by a path of F nodes, as (grid, j, i, sign), in the order of the path.
    sign: multiplier such that the flux is positive from the left to the right of the path
    '''
    edges = []
    for (ja, ia), (jb, ib) in zip(nodes[:-1], nodes[1:]):
        if ja == jb:    # step in i, along a V face (north face of T[j, max(i)])
            edges.append(("V", ja, max(ia, ib), -(ib - ia)))
        else:           # step in j, along a U face (east face of T[max(j), i])
            edges.append(("U", max(ja, jb), ia, jb - ja))
    return edges


def edges_to_points(edges, idxmap):
    '''
    bounmask indexes of the water points (all levels) that own the faces of the transect, 
    with sign multipliers U, V, W.
    The faces on land (no water level in the owner cell) and on the border of the domain 
    (as in the other control surfaces) are skipped.
    '''
    jpk, jpj, jpi = idxmap.shape
    faces = {}
    n_land = 0
    for grid, j, i, sign in edges:
        if j < 1 or j > jpj - 2 or i < 1 or i > jpi - 2:
            continue
        if not idxmap[:, j, i].any():
            n_land += 1
            continue
        if faces.get((grid, j, i), sign) != sign:
            raise ValueError(f"the transect crosses the face {grid}[{j},{i}] in both directions, "
                             "the sign convention is not defined")
        faces[(grid, j, i)] = sign
    index, p_U, p_V = [], [], []
    position = {}
    for (grid, j, i), sign in faces.items():
        for x in idxmap[:, j, i]:
            if x == 0:
                continue
            if x not in position:
                position[x] = len(index)
                index.append(int(x))
                p_U.append(0)
                p_V.append(0)
            (p_U if grid == "U" else p_V)[position[x]] = sign
    p_W = [0] * len(index)
    return index, p_U, p_V, p_W, n_land


def compute_path(points, glamf, gphif, verbose=False):
    '''
    path on the F lattice through the points, as ordered F nodes (j, i) 
    and faces 'edges' (grid, j, i, sign). No information on the land-sea mask is needed.
    '''
    if len(points) < 2:
        raise ValueError("an arbitrary transect needs at least two points")
    snapped = []
    for lon, lat in points:
        node = snap_to_node(float(lon), float(lat), glamf, gphif)
        if verbose:
            print(f"  point ({lon}, {lat}) -> F node j={node[0]}, i={node[1]} "
                  f"({glamf[node]:.3f}, {gphif[node]:.3f})")
        if not snapped or node != snapped[-1]:
            snapped.append(node)
    if len(snapped) < 2:
        raise ValueError("all the points of the transect fall on the same grid node")
    nodes = [snapped[0]]
    for n0, n1 in zip(snapped[:-1], snapped[1:]):
        nodes += staircase(n0, n1, glamf, gphif)[1:]
    return nodes, nodes_to_edges(nodes)


def compute_arbitrary_transect(points, glamf, gphif, idxmap, verbose=False):
    '''
    points: list of (lon, lat), at least two
    glamf, gphif: F-point coordinates, (jpj, jpi)
    idxmap: bounmask index of the water points, 0 on land, (jpk, jpj, jpi)

    Returns a dictionary with the control surface:
    index, U, V, W (lists, one value per water point of the transect), 
    and the ordered F 'nodes' (j, i) and 'edges' (grid, j, i, sign) of the path.
    '''
    nodes, edges = compute_path(points, glamf, gphif, verbose)
    index, p_U, p_V, p_W, n_land = edges_to_points(edges, idxmap)
    if verbose:
        n_u = sum(e[0] == "U" for e in edges)
        print(f"  {len(edges)} faces (U={n_u}, V={len(edges) - n_u}), "
              f"{n_land} on land, {len(index)} water points")
    if len(index) == 0:
        raise ValueError("the transect does not cross any water point")
    return {"index": index, "U": p_U, "V": p_V, "W": p_W, "nodes": nodes, "edges": edges}
