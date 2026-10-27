import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

# We'll import the app inside the test functions after patching the model file existence
# to avoid the FileNotFoundError during module import.

def test_health_endpoint():
    """Test the health check endpoint."""
    # We need to mock the model file existence so that the analyzer can be initialized without error.
    with patch('backend.vision.detector.os.path.exists', return_value=True):
        # We also need to mock the cv2.dnn.readNetFromONNX to avoid trying to load a real model.
        with patch('cv2.dnn.readNetFromONNX') as mock_read_net:
            mock_net = MagicMock()
            mock_net.getUnconnectedOutLayersNames.return_value = ['output0']
            mock_read_net.return_value = mock_net

            # Now we can import the app
            from backend.main import app
            client = TestClient(app)

            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
            assert "timestamp" in data

def test_list_incidents_empty():
    """Test listing incidents when there are none."""
    with patch('backend.vision.detector.os.path.exists', return_value=True):
        with patch('cv2.dnn.readNetFromONNX') as mock_read_net:
            mock_net = MagicMock()
            mock_net.getUnconnectedOutLayersNames.return_value = ['output0']
            mock_read_net.return_value = mock_net

            from backend.main import app
            client = TestClient(app)

            # Mock the incidents directory to be empty
            with patch('backend.main.Path') as mock_path_class:
                # We need to mock the get_incident_dirs function to return an empty list.
                # The get_incident_dirs function uses Path("incidents").iterdir()
                # We'll mock the Path constructor for the incidents directory.
                # Instead, we can directly mock the get_incident_dirs function.
                with patch('backend.main.list_incidents_filtered', return_value=[]):
                    response = client.get("/incidents")
                    assert response.status_code == 200
                    data = response.json()
                    assert data == []

def test_get_incident_not_found():
    """Test getting a non-existent incident."""
    with patch('backend.vision.detector.os.path.exists', return_value=True):
        with patch('cv2.dnn.readNetFromONNX') as mock_read_net:
            mock_net = MagicMock()
            mock_net.getUnconnectedOutLayersNames.return_value = ['output0']
            mock_read_net.return_value = mock_net

            from backend.main import app
            client = TestClient(app)

            with patch('backend.main.load_incident_metadata', return_value=None):
                response = client.get("/incidents/nonexistent")
                assert response.status_code == 404

def test_app_can_be_imported():
    """Test that the app can be imported without raising FileNotFoundError for the model."""
    with patch('backend.vision.detector.os.path.exists', return_value=True):
        with patch('cv2.dnn.readNetFromONNX') as mock_read_net:
            mock_net = MagicMock()
            mock_net.getUnconnectedOutLayersNames.return_value = ['output0']
            mock_read_net.return_value = mock_net

            # This import should not raise an exception
            from backend.main import app
            assert app is not None

if __name__ == "__main__":
    pytest.main([__file__, "-v"])