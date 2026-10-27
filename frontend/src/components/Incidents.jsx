import React, { useState, useEffect } from 'react';
import axios from 'axios';
import IncidentDetail from './IncidentDetail';

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

const Incidents = () => {
  const [incidents, setIncidents] = useState([]);
  const [selectedIncidentId, setSelectedIncidentId] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Filters
  const [statusFilter, setStatusFilter] = useState('');
  const [severityFilter, setSeverityFilter] = useState('');
  const [searchQuery, setSearchQuery] = useState('');

  useEffect(() => {
    const fetchIncidents = async () => {
      try {
        const params = new URLSearchParams();
        if (statusFilter) params.append('status', statusFilter);
        if (severityFilter) params.append('severity', severityFilter);

        const response = await axios.get(`${API_URL}/incidents`, {
          params,
          timeout: 10000 // 10 second timeout for incident listing
        });
        setIncidents(response.data);
        setLoading(false);
      } catch (err) {
        console.error(err);
        if (err.code === 'ECONNABORTED' || err.message?.includes('timeout')) {
          setError('Request timed out. The backend may be overloaded.');
        } else if (err.response) {
          setError(`Server error (${err.response.status}): Failed to load incidents.`);
        } else if (err.request) {
          setError('No response from the backend. Check that the API is running at ' + API_URL + '.');
        } else {
          setError('An unexpected error occurred while loading incidents.');
        }
        setLoading(false);
      }
    };

    fetchIncidents();
    const interval = setInterval(fetchIncidents, 5000); // refresh every 5 seconds
    return () => clearInterval(interval);
  }, [statusFilter, severityFilter]);

  const handleApprove = async (incidentId) => {
    try {
      await axios.post(`${API_URL}/incidents/${incidentId}/approve`, { status: 'approved' }, { timeout: 10000 });
      // Refresh incidents list
      const response = await axios.get(`${API_URL}/incidents`, { timeout: 10000 });
      setIncidents(response.data);
      setSelectedIncidentId(null);
    } catch (err) {
      console.error(err);
      if (err.code === 'ECONNABORTED' || err.message?.includes('timeout')) {
        alert('Request timed out. Please try again.');
      } else if (err.response) {
        alert(`Failed to approve incident: ${err.response.status} ${err.response.statusText}`);
      } else if (err.request) {
        alert('No response from the backend. Check that the API is running.');
      } else {
        alert('An unexpected error occurred.');
      }
    }
  };

  const handleReject = async (incidentId) => {
    try {
      await axios.post(`${API_URL}/incidents/${incidentId}/reject`, { status: 'rejected' }, { timeout: 10000 });
      // Refresh incidents list
      const response = await axios.get(`${API_URL}/incidents`, { timeout: 10000 });
      setIncidents(response.data);
      setSelectedIncidentId(null);
    } catch (err) {
      console.error(err);
      if (err.code === 'ECONNABORTED' || err.message?.includes('timeout')) {
        alert('Request timed out. Please try again.');
      } else if (err.response) {
        alert(`Failed to reject incident: ${err.response.status} ${err.response.statusText}`);
      } else if (err.request) {
        alert('No response from the backend. Check that the API is running.');
      } else {
        alert('An unexpected error occurred.');
      }
    }
  };

  // Filter incidents based on search query
  const filteredIncidents = incidents.filter(inc => {
    const matchesSearch = !searchQuery ||
      inc.incident_id.toLowerCase().includes(searchQuery.toLowerCase()) ||
      (inc.event_type && inc.event_type.toLowerCase().includes(searchQuery.toLowerCase())) ||
      (inc.zone && inc.zone.toLowerCase().includes(searchQuery.toLowerCase()));
    return matchesSearch;
  });

  const getSeverityClass = (severity) => {
    switch (severity) {
      case 'critical': return 'severity-critical';
      case 'high': return 'severity-high';
      case 'medium': return 'severity-medium';
      case 'low': return 'severity-low';
      default: return '';
    }
  };

  const getStatusClass = (status) => {
    switch (status) {
      case 'approved': return 'status-approved';
      case 'rejected': return 'status-rejected';
      case 'confirmed': return 'status-confirmed';
      case 'under_review': return 'status-under-review';
      case 'candidate': return 'status-candidate';
      case 'resolved': return 'status-resolved';
      default: return '';
    }
  };

  const statusOptions = [
    { value: '', label: 'All Statuses' },
    { value: 'candidate', label: 'Candidate' },
    { value: 'confirmed', label: 'Confirmed' },
    { value: 'under_review', label: 'Under Review' },
    { value: 'approved', label: 'Approved' },
    { value: 'rejected', label: 'Rejected' },
    { value: 'resolved', label: 'Resolved' }
  ];

  const severityOptions = [
    { value: '', label: 'All Severities' },
    { value: 'critical', label: 'Critical' },
    { value: 'high', label: 'High' },
    { value: 'medium', label: 'Medium' },
    { value: 'low', label: 'Low' }
  ];

  if (loading) {
    return <div className="incidents">Loading incidents...</div>;
  }

  if (error) {
    return <div className="incidents error">{error}</div>;
  }

  return (
    <div className="incidents">
      <div className="incidents-header">
        <h2>Incidents</h2>
        <div className="incident-stats">
          <span>Total: {incidents.length}</span>
          <span>Showing: {filteredIncidents.length}</span>
        </div>
      </div>

      {/* Filters */}
      <div className="filters-bar">
        <div className="search-box">
          <input
            type="text"
            placeholder="Search incidents..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>
        <div className="filter-group">
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="filter-select"
          >
            {statusOptions.map(opt => (
              <option key={opt.value} value={opt.value}>{opt.label}</option>
            ))}
          </select>
          <select
            value={severityFilter}
            onChange={(e) => setSeverityFilter(e.target.value)}
            className="filter-select"
          >
            {severityOptions.map(opt => (
              <option key={opt.value} value={opt.value}>{opt.label}</option>
            ))}
          </select>
          {(statusFilter || severityFilter) && (
            <button
              onClick={() => { setStatusFilter(''); setSeverityFilter(''); }}
              className="clear-filters"
            >
              Clear Filters
            </button>
          )}
        </div>
      </div>

      {selectedIncidentId ? (
        <IncidentDetail
          incidentId={selectedIncidentId}
          onClose={() => setSelectedIncidentId(null)}
          onApprove={handleApprove}
          onReject={handleReject}
        />
      ) : (
        <>
          <ul className="incident-list">
            {filteredIncidents.length === 0 ? (
              <li className="no-incidents">No incidents found matching your criteria.</li>
            ) : (
              filteredIncidents.map((inc) => (
                <li key={inc.incident_id} className="incident-item">
                  <div className="incident-main">
                    <span className="incident-id">{inc.incident_id.slice(0, 8)}...</span>
                    <span className="incident-event">{inc.event_type || 'Unknown'}</span>
                    <span className="incident-time">
                      {inc.timestamp ? new Date(inc.timestamp * 1000).toLocaleString() : 'Unknown time'}
                    </span>
                    <span className={`incident-status ${getStatusClass(inc.status)}`}>
                      {inc.status || 'Unknown'}
                    </span>
                    {inc.severity && (
                      <span className={`severity-badge ${getSeverityClass(inc.severity)}`}>
                        {inc.severity.toUpperCase()}
                      </span>
                    )}
                  </div>
                  <div className="incident-secondary">
                    {inc.zone && <span className="incident-zone">Zone: {inc.zone}</span>}
                    {inc.confidence !== undefined && (
                      <span className="incident-confidence">
                        Confidence: {(inc.confidence * 100).toFixed(1)}%
                      </span>
                    )}
                    {inc.duration !== undefined && (
                      <span className="incident-duration">
                        Duration: {inc.duration.toFixed(1)}s
                      </span>
                    )}
                    {inc.storage_backend && (
                      <span className="incident-storage">
                        Storage: {inc.storage_backend.toUpperCase()}
                      </span>
                    )}
                  </div>
                  <button
                    onClick={() => setSelectedIncidentId(inc.incident_id)}
                    className="view-details-btn"
                  >
                    View Details
                  </button>
                </li>
              ))
            )}
          </ul>
        </>
      )}
    </div>
  );
};

export default Incidents;