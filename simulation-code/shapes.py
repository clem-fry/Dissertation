import numpy as np
from functools import lru_cache

mask_nodes = None

def grid_check(r, coords):
    grid_size = r*2
    if np.ndim(coords) == 1:
        x = coords[0]  # Shape will become (1, 2)
        y = coords[1]
        return True
    #(x != 1 or x != grid_size or y != 1 or y != grid_size)
    else:
        x = coords[:, 0]
        y = coords[:, 1]
        return True
    #(x != 1) | (x != grid_size) | (y != 1) | (y != grid_size)
    
def small_grid_check(r, coords):
    inner_min = r // 2
    inner_max = 3 * r // 2

    if np.ndim(coords) == 1:
        x, y = coords
        return inner_min <= x < inner_max and inner_min <= y < inner_max
    else:
        x = coords[:, 0]
        y = coords[:, 1]
        return (inner_min <= x) & (x < inner_max) & (inner_min <= y) & (y < inner_max)

def circle_outline_check(r, coords):
    if np.ndim(coords) == 1:
        x = coords[0]  # Shape will become (1, 2)
        y = coords[1]
    else:
        x = coords[:, 0]
        y = coords[:, 1]

    # is the node in the circle?
    return (np.square(np.array(x) - r) + np.square(np.array(y) - r)) <= r ** 2

def heart_outline_check(r, coords):
    if np.ndim(coords) == 1:
        x = coords[0]  # Shape will become (1, 2)
        y = coords[1]
    else:
        x = coords[:, 0]
        y = coords[:, 1]

    s = r * 0.87
    x_scaled = (x - 1.15 * s) / s
    y_scaled = (y - 1 * s) / s

    # is the node in the circle?
    return ((x_scaled**2 + y_scaled**2 - 1)**3 - x_scaled**2 * y_scaled**3) <= 0


def triangle_outline_check(r, coords):
    coords = np.atleast_2d(coords)
    x = coords[:, 0] -1
    y = coords[:, 1] -1

    # Conditions:
    # - inside vertical side: 0 ≤ y ≤ r
    # - inside horizontal side: 0 ≤ x ≤ r
    # - below or on hypotenuse: x + y ≤ r
    return (x >= 0) & (y >= 0) & (x <= r) & (y <= r) & (x + y <= r)

def ring_check(r, coords, inner=0.5):
    # annulus centred in the [1, 2r] square: outer radius r, inner radius inner * r
    coords = np.atleast_2d(coords)
    x = coords[:, 0] - (r + 0.5)
    y = coords[:, 1] - (r + 0.5)
    distance = np.hypot(x, y)
    return (distance <= r) & (distance >= inner * r)

def horseshoe_check(r, coords, inner=0.5, gap=60):
    # ring with an opening of `gap` degrees at the bottom
    coords = np.atleast_2d(coords)
    x = coords[:, 0] - (r + 0.5)
    y = coords[:, 1] - (r + 0.5)
    angle_from_bottom = np.degrees(np.abs(np.arctan2(x, -y)))  # 0 straight down, 180 straight up
    return ring_check(r, coords, inner) & (angle_from_bottom >= gap / 2)

@lru_cache(maxsize=None)
def _circle_cells(r):
    # one-cell-thick loop of lattice cells round a circle of radius r - 0.5 centred in the [1, 2r] square
    centre, radius = r + 0.5, r - 0.5
    angles = np.linspace(0, 2 * np.pi, int(8 * np.pi * radius), endpoint=False)
    cells = np.unique(np.round(centre + radius * np.column_stack([np.cos(angles), np.sin(angles)])), axis=0)
    cells = cells[np.argsort(np.arctan2(cells[:, 1] - centre, cells[:, 0] - centre))]
    # drop corner cells whose two neighbours along the loop already touch (diagonally), so every cell has
    # exactly two neighbours along the loop
    changed = True
    while changed:
        changed = False
        for k in range(len(cells)):
            before, after = cells[k - 1], cells[(k + 1) % len(cells)]
            if np.abs(before - after).max() <= 1:
                cells = np.delete(cells, k, axis=0)
                changed = True
                break
    return frozenset(map(tuple, cells.astype(int)))

def ring_line_check(r, coords):
    # ring one node wide: run_turing links each node only to the next one round the loop (chain = 'closed')
    coords = np.atleast_2d(coords)
    cells = _circle_cells(r)
    return np.array([(int(x), int(y)) in cells for x, y in coords])

ring_line_check.chain = 'closed'

def horseshoe_line_check(r, coords, gap=60):
    # ring_line_check with an opening of `gap` degrees at the bottom: an open chain with two free ends
    coords = np.atleast_2d(coords)
    x = coords[:, 0] - (r + 0.5)
    y = coords[:, 1] - (r + 0.5)
    angle_from_bottom = np.degrees(np.abs(np.arctan2(x, -y)))
    return ring_line_check(r, coords) & (angle_from_bottom >= gap / 2)

horseshoe_line_check.chain = 'open'

def line_check(r, coords):
    coords = np.atleast_2d(coords)
    x = coords[:, 0]
    y = coords[:, 1]
    return (y == 1)

def point_number(r, coords):
    # points = (grid_size / 2)**2
    coords = np.atleast_2d(coords)

    r = int(r*2)
    # Generate r integer grid points laid out in a compact grid (e.g., row-first)
    rows = int(np.floor(np.sqrt(r)))
    cols = int(np.ceil(r / rows))

    grid_points = [(i, j) for i in range(rows) for j in range(cols)]
    grid_points = np.array(grid_points[:r], dtype=int) + 1

    # Check if each input coord matches one of the grid points
    return np.any((coords[:, None] == grid_points).all(-1), axis=1)