import open3d as o3d
import numpy as np
from typing import List, Tuple, Optional


def create_arrow_p1_to_p2(
    p1: List[float],
    p2: List[float],
    color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    shaft_radius: float = 0.02,
    head_radius: float = 0.04,
    head_length: float = 0.08,
    shaft_length_ratio: float = 0.8
) -> o3d.geometry.TriangleMesh:
    """
    Create an arrow pointing from p1 to p2.

    Args:
        p1: Starting point [x, y, z]
        p2: Ending point [x, y, z]
        color: RGB color of the arrow (default: red)
        shaft_radius: Radius of the arrow shaft
        head_radius: Radius of the arrow head
        head_length: Length of the arrow head
        shaft_length_ratio: Ratio of shaft length to total arrow length

    Returns:
        o3d.geometry.TriangleMesh: Arrow mesh pointing from p1 to p2
    """

    # Convert to numpy arrays
    p1 = np.array(p1, dtype=np.float64)
    p2 = np.array(p2, dtype=np.float64)

    # Calculate direction and total length
    direction = p2 - p1
    total_length = np.linalg.norm(direction)

    if total_length < 1e-6:
        raise ValueError("Points p1 and p2 are too close together")

    # Normalize direction
    direction_normalized = direction / total_length

    # Calculate shaft length
    shaft_length = total_length * shaft_length_ratio

    # Create shaft (cylinder)
    shaft = o3d.geometry.TriangleMesh.create_cylinder(
        radius=shaft_radius,
        height=shaft_length,
        resolution=20
    )

    # Create head (cone)
    head = o3d.geometry.TriangleMesh.create_cone(
        radius=head_radius,
        height=head_length,
        resolution=20
    )

    # Position the shaft at the start
    shaft.translate([0, 0, shaft_length / 2])

    # Position the head at the end of the shaft
    head.translate([0, 0, shaft_length])

    # Combine shaft and head
    arrow = shaft + head

    # Rotate arrow to point in the right direction
    # Default arrow points along z-axis, so we need to rotate it
    z_axis = np.array([0, 0, 1])

    if not np.allclose(direction_normalized, z_axis):
        # Calculate rotation axis and angle
        rotation_axis = np.cross(z_axis, direction_normalized)
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        cos_angle = np.dot(z_axis, direction_normalized)
        angle = np.arccos(np.clip(cos_angle, -1, 1))

        # Apply rotation
        R = arrow.get_rotation_matrix_from_axis_angle(rotation_axis * angle)
        arrow.rotate(R, center=[0, 0, 0])

    # Move arrow to start at p1
    arrow.translate(p1)

    # Set color
    arrow.paint_uniform_color(color)

    return arrow


def create_arrow_p1_to_p2_simple(
    p1: List[float],
    p2: List[float],
    color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    scale: float = 1.0
) -> o3d.geometry.TriangleMesh:
    """
    Create a simple arrow pointing from p1 to p2 using Open3D's built-in create_arrow.

    Args:
        p1: Starting point [x, y, z]
        p2: Ending point [x, y, z]
        color: RGB color of the arrow (default: red)
        scale: Scale factor for the arrow size

    Returns:
        o3d.geometry.TriangleMesh: Arrow mesh pointing from p1 to p2
    """

    # Convert to numpy arrays
    p1 = np.array(p1, dtype=np.float64)
    p2 = np.array(p2, dtype=np.float64)

    # Calculate direction and length
    direction = p2 - p1
    length = np.linalg.norm(direction)

    if length < 1e-6:
        raise ValueError("Points p1 and p2 are too close together")

    # Normalize direction
    direction_normalized = direction / length

    # Create arrow using Open3D's built-in function
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.02 * scale,
        cone_radius=0.04 * scale,
        cylinder_height=length * 0.8,  # 80% of total length for shaft
        cone_height=length * 0.2,      # 20% of total length for head
        resolution=20,
        cylinder_split=4,
        cone_split=1
    )

    # Rotate arrow to point in the right direction
    z_axis = np.array([0, 0, 1])

    if not np.allclose(direction_normalized, z_axis):
        # Calculate rotation axis and angle
        rotation_axis = np.cross(z_axis, direction_normalized)
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        cos_angle = np.dot(z_axis, direction_normalized)
        angle = np.arccos(np.clip(cos_angle, -1, 1))

        # Apply rotation
        R = arrow.get_rotation_matrix_from_axis_angle(rotation_axis * angle)
        arrow.rotate(R, center=[0, 0, 0])

    # Move arrow to start at p1
    arrow.translate(p1)

    # Set color
    arrow.paint_uniform_color(color)

    return arrow


def create_multiple_arrows(
    arrow_data: List[Tuple[List[float], List[float], Tuple[float, float, float]]],
    scale: float = 1.0
) -> List[o3d.geometry.TriangleMesh]:
    """
    Create multiple arrows from a list of (p1, p2, color) tuples.

    Args:
        arrow_data: List of tuples (p1, p2, color) where:
                   p1: starting point [x, y, z]
                   p2: ending point [x, y, z]
                   color: RGB color tuple
        scale: Scale factor for all arrows

    Returns:
        List[o3d.geometry.TriangleMesh]: List of arrow meshes
    """

    arrows = []
    for p1, p2, color in arrow_data:
        arrow = create_arrow_p1_to_p2_simple(p1, p2, color, scale)
        arrows.append(arrow)

    return arrows


# Example usage and test function
def test_arrow_functions():
    """
    Test function to demonstrate arrow creation.
    """

    # Test points
    p1 = [0, 0, 0]
    p2 = [1, 1, 1]

    # Create arrow using custom function
    arrow1 = create_arrow_p1_to_p2(p1, p2, color=(1.0, 0.0, 0.0))

    # Create arrow using simple function
    arrow2 = create_arrow_p1_to_p2_simple(p1, p2, color=(0.0, 1.0, 0.0))

    # Create multiple arrows
    arrow_data = [
        ([0, 0, 0], [1, 0, 0], (1.0, 0.0, 0.0)),  # Red arrow along X
        ([0, 0, 0], [0, 1, 0], (0.0, 1.0, 0.0)),  # Green arrow along Y
        ([0, 0, 0], [0, 0, 1], (0.0, 0.0, 1.0)),  # Blue arrow along Z
    ]
    arrows = create_multiple_arrows(arrow_data)

    # Add coordinate frame for reference
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.3)

    # Visualize
    geometries = [arrow1, arrow2] + arrows + [coordinate_frame]
    o3d.visualization.draw_geometries(geometries)


if __name__ == "__main__":
    test_arrow_functions()
