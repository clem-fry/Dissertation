import numpy as np

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

def line_check(r, coords):
    coords = np.atleast_2d(coords)
    x = coords[:, 0]
    y = coords[:, 1]
    return (y == 1)

def point_number(r, coords):
    coords = np.atleast_2d(coords)

    r = int(r**2)
    # Generate r integer grid points laid out in a compact grid (e.g., row-first)
    rows = int(np.floor(np.sqrt(r)))
    cols = int(np.ceil(r / rows))

    grid_points = [(i, j) for i in range(rows) for j in range(cols)]
    grid_points = np.array(grid_points[:r], dtype=int)

    # Check if each input coord matches one of the grid points
    return np.any((coords[:, None] == grid_points).all(-1), axis=1)