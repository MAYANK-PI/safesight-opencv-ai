import numpy as np
from backend.vision.roi import RegionOfInterest

def test_roi_point_in_polygon():
    """Test point-in-polygon for a simple square."""
    # Define a square polygon from (10,10) to (20,20)
    polygon = [(10, 10), (20, 10), (20, 20), (10, 20)]
    roi = RegionOfInterest(polygon)

    # Points inside
    assert roi.is_inside((15, 15)) == True
    assert roi.is_inside((10, 10)) == True  # on edge, consider inside
    assert roi.is_inside((20, 20)) == True  # on edge

    # Points outside
    assert roi.is_inside((5, 5)) == False
    assert roi.is_inside((25, 25)) == False
    assert roi.is_inside((15, 5)) == False
    assert roi.is_inside((5, 15)) == False

def test_roi_centroid():
    """Test centroid calculation for a rectangle."""
    polygon = [(10, 10), (20, 10), (20, 20), (10, 20)]
    roi = RegionOfInterest(polygon)
    centroid = roi.get_centroid()
    # Expected centroid of a rectangle is (15,15)
    assert centroid == (15, 15)

    # Test with a non-rectangular polygon
    polygon2 = [(0, 0), (10, 0), (10, 10), (5, 15), (0, 10)]
    roi2 = RegionOfInterest(polygon2)
    centroid2 = roi2.get_centroid()
    # We won't check the exact value, just that it returns a tuple of ints
    assert isinstance(centroid2, tuple)
    assert len(centroid2) == 2
    assert isinstance(centroid2[0], int)
    assert isinstance(centroid2[1], int)

def test_roi_draw():
    """Test drawing the ROI on a frame."""
    polygon = [(10, 10), (20, 10), (20, 20), (10, 20)]
    roi = RegionOfInterest(polygon)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    drawn = roi.draw(frame, color=(0, 255, 0), thickness=2)
    # The drawn frame should not be all zeros anymore
    assert np.any(drawn != 0)