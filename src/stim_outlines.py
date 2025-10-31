import numpy as np
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union

# -------------------------------------------------------------------------------
def movingBar(
        lEdge: int = 300, trajLen :int = 4000,
        angles: list = [0, 45, 90, 135],
        x0: int = 0, 
        y0: int = 0,
        FOV_diam: int = None
    ):

    def _create_rotated_rect(w, h, a, cx, cy):
        # Create rectangle centered at origin
        half_w = w / 2
        half_h = h / 2

        # Define corner points
        corners = np.array([
            [-half_w, -half_h],
            [half_w, -half_h],
            [half_w, half_h],
            [-half_w, half_h]
        ])

        # Rotate
        angle_rad = np.radians(a)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

        rotated_corners = corners @ rotation_matrix.T

        # Translate to center
        rotated_corners[:, 0] += cx
        rotated_corners[:, 1] += cy
        return Polygon(rotated_corners)


    # Create all rectangles
    rectangles = []
    for angle in angles:
        rect = _create_rotated_rect(lEdge, trajLen, angle, x0, y0)
        rectangles.append(rect)

    # Create union of all rectangles to form the final polygon
    union_polygon = unary_union(rectangles)

    # Apply FOV aperture if specified
    if FOV_diam is not None:
        aperture = Point(x0, y0).buffer(FOV_diam / 2)
        return union_polygon.intersection(aperture)

    return union_polygon

# -------------------------------------------------------------------------------
def spot(
        diam: int = 1000,
        x0: int = 0,
        y0: int = 0,
        FOV_diam: int = None
    ):
    # Create circle centered at (x0, y0)
    circle_polygon = Point(x0, y0).buffer(diam / 2)

    # Apply FOV aperture if specified
    if FOV_diam is not None:
        aperture = Point(x0, y0).buffer(FOV_diam / 2)
        return circle_polygon.intersection(aperture)

    return circle_polygon

# -------------------------------------------------------------------------------
def box(
        dx: int = 1000,
        dy: int = 1000,
        angle: float = 0,
        x0: int = 0,
        y0: int = 0,
        FOV_diam: int = None
    ):
    # Create rectangle centered at origin
    half_w = dx / 2
    half_h = dy / 2

    # Define corner points
    corners = np.array([
        [-half_w, -half_h],
        [half_w, -half_h],
        [half_w, half_h],
        [-half_w, half_h]
    ])

    # Rotate
    angle_rad = np.radians(angle)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    rotated_corners = corners @ rotation_matrix.T

    # Translate to center
    rotated_corners[:, 0] += x0
    rotated_corners[:, 1] += y0

    # Create polygon
    rect_polygon = Polygon(rotated_corners)

    # Apply FOV aperture if specified
    if FOV_diam is not None:
        aperture = Point(x0, y0).buffer(FOV_diam / 2)
        return rect_polygon.intersection(aperture)

    return rect_polygon

# -------------------------------------------------------------------------------
