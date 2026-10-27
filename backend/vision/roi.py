import cv2
import numpy as np
from typing import List, Tuple

class RegionOfInterest:
    def __init__(self, polygon: List[Tuple[int, int]]):
        """
        Initialize the region of interest with a polygon.

        Args:
            polygon: List of (x, y) tuples defining the polygon vertices.
        """
        self.polygon = np.array(polygon, dtype=np.int32)
        # For point-in-polygon test, we need the polygon in the format expected by cv2.pointPolygonTest
        # which is a single-channel array of points.

    def is_inside(self, point: Tuple[int, int]) -> bool:
        """
        Check if a point is inside the polygon.

        Args:
            point: (x, y) tuple.

        Returns:
            True if the point is inside the polygon, False otherwise.
        """
        # Convert point to float for cv2.pointPolygonTest
        point_float = (float(point[0]), float(point[1]))
        # cv2.pointPolygonTest returns positive if inside, negative if outside, zero if on edge
        result = cv2.pointPolygonTest(self.polygon, point_float, False)
        return result >= 0

    def get_centroid(self) -> Tuple[int, int]:
        """
        Calculate the centroid of the polygon.

        Returns:
            (x, y) centroid coordinates.
        """
        # Using the formula for centroid of a polygon
        # We'll compute using the vertices
        area = 0.0
        centroid_x = 0.0
        centroid_y = 0.0
        n = len(self.polygon)
        for i in range(n):
            x0, y0 = self.polygon[i]
            x1, y1 = self.polygon[(i + 1) % n]
            cross = x0 * y1 - x1 * y0
            area += cross
            centroid_x += (x0 + x1) * cross
            centroid_y += (y0 + y1) * cross
        area *= 0.5
        if area == 0:
            # Fallback to average of vertices
            centroid_x = np.mean(self.polygon[:, 0])
            centroid_y = np.mean(self.polygon[:, 1])
        else:
            centroid_x /= (6.0 * area)
            centroid_y /= (6.0 * area)
        return int(centroid_x), int(centroid_y)

    def draw(self, frame: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2) -> np.ndarray:
        """
        Draw the polygon on the frame.

        Args:
            frame: Input image (BGR).
            color: BGR color tuple.
            thickness: Line thickness.

        Returns:
            The frame with the polygon drawn.
        """
        # Make a copy to avoid modifying the original if not intended
        output = frame.copy()
        # Reshape polygon for cv2.polylines
        pts = self.polygon.reshape((-1, 1, 2))
        cv2.polylines(output, [pts], isClosed=True, color=color, thickness=thickness)
        return output