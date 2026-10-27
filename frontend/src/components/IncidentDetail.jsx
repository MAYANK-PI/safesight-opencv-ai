import React, { useState, useEffect } from 'react';
import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

const IncidentDetail = ({ incidentId, onClose, onApprove, onReject }) => {
  const [incident, setIncident] = useState(null);
  const [timeline, setTimeline] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchIncident = async () => {
      try {
        const response = await axios.get(`${API_URL}/incidents/${incidentId}`, { timeout: 10000 });
        setIncident(response.data);
        setLoading(false);

        // Also fetch timeline
        const timelineResponse = await axios.get(`${API_URL}/incidents/${incidentId}/timeline`, { timeout: 10000 });
        setTimeline(timelineResponse.data || []);
      } catch (err) {
        console.error(err);
        if (err.code === 'ECONNABORTED' || err.message?.includes('timeout')) {
          setError('Request timed out. The backend may be overloaded.');
        } else if (err.response) {
          setError(`Server error (${err.response.status}): Failed to load incident details.`);
        } else if (err.request) {
          setError('No response from the backend. Check that the API is running at ' + API_URL + '.');
        } else {
          setError('An unexpected error occurred while loading incident details.');
        }
        setLoading(false);
      }
    };

    fetchIncident();
  }, [incidentId]);

  if (loading) {
    return <div className="incident-detail">Loading incident details...</div>;
  }

  if (error) {
    return <div className="incident-detail error">{error}</div>;
  }

  if (!incident) {
    return <div className="incident-detail">No incident data.</div>;
  }

  const evidenceUrl = `${API_URL}/incidents/${incidentId}/evidence.jpg`;

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

  const formatTimestamp = (ts) => {
    if (!ts) return 'N/A';
    return new Date(ts * 1000).toLocaleString();
  };

  return (
    <div className="incident-detail">
      <div className="header">
        <h2>Incident Details</h2>
        <button onClick={onClose} className="close-button">×</button>
      </div>
      <div className="content">
        {/* Header with key info */}
        <div className="incident-header">
          <div className="incident-id-full">{incident.incident_id}</div>
          <div className="incident-badges">
            <span className={`severity-badge ${getSeverityClass(incident.severity)}`}>
              {incident.severity?.toUpperCase() || 'UNKNOWN'}
            </span>
            <span className={`incident-status ${getStatusClass(incident.status)}`}>
              {incident.status || 'UNKNOWN'}
            </span>
          </div>
        </div>

        {/* Metadata Grid */}
        <div className="metadata-grid">
          <div className="metadata-section">
            <h3>Basic Information</h3>
            <div className="metadata-item">
              <span className="metadata-label">Event Type</span>
              <span className="metadata-value">{incident.event_type || 'N/A'}</span>
            </div>
            <div className="metadata-item">
              <span className="metadata-label">Person ID</span>
              <span className="metadata-value">{incident.person_id || 'N/A'}</span>
            </div>
            <div className="metadata-item">
              <span className="metadata-label">Zone</span>
              <span className="metadata-value">{incident.zone || 'N/A'}</span>
            </div>
            <div className="metadata-item">
              <span className="metadata-label">Confidence</span>
              <span className="metadata-value">
                {incident.confidence !== undefined ? `${(incident.confidence * 100).toFixed(1)}%` : 'N/A'}
              </span>
            </div>
            <div className="metadata-item">
              <span className="metadata-label">Duration</span>
              <span className="metadata-value">
                {incident.duration !== undefined ? `${incident.duration.toFixed(2)}s` : 'N/A'}
              </span>
            </div>
            <div className="metadata-item">
              <span className="metadata-label">Risk Level</span>
              <span className="metadata-value">{incident.risk_level || 'N/A'}</span>
            </div>
          </div>

          <div className="metadata-section">
            <h3>Timestamps</h3>
            <div className="metadata-item">
              <span className="metadata-label">Created</span>
              <span className="metadata-value">{formatTimestamp(incident.timestamp)}</span>
            </div>
            <div className="metadata-item">
              <span className="metadata-label">Updated</span>
              <span className="metadata-value">{formatTimestamp(incident.updated_at)}</span>
            </div>
            {incident.approved_at && (
              <div className="metadata-item">
                <span className="metadata-label">Approved</span>
                <span className="metadata-value">{formatTimestamp(incident.approved_at)}</span>
              </div>
            )}
            {incident.rejected_at && (
              <div className="metadata-item">
                <span className="metadata-label">Rejected</span>
                <span className="metadata-value">{formatTimestamp(incident.rejected_at)}</span>
              </div>
            )}
            {incident.resolved_at && (
              <div className="metadata-item">
                <span className="metadata-label">Resolved</span>
                <span className="metadata-value">{formatTimestamp(incident.resolved_at)}</span>
              </div>
            )}
          </div>

          {/* Storage info */}
          {(incident.storage_backend || incident.evidence_path_s3 || incident.metadata_path_s3) && (
            <div className="metadata-section">
              <h3>Storage</h3>
              <div className="metadata-item">
                <span className="metadata-label">Primary Backend</span>
                <span className="metadata-value">{incident.storage_backend?.toUpperCase() || 'LOCAL'}</span>
              </div>
              {incident.evidence_path_s3 && (
                <div className="metadata-item">
                  <span className="metadata-label">Evidence (S3)</span>
                  <span className="metadata-value">
                    <a href={incident.evidence_path_s3} target="_blank" rel="noopener noreferrer">
                      {incident.evidence_path_s3}
                    </a>
                  </span>
                </div>
              )}
              {incident.metadata_path_s3 && (
                <div className="metadata-item">
                  <span className="metadata-label">Metadata (S3)</span>
                  <span className="metadata-value">
                    <a href={incident.metadata_path_s3} target="_blank" rel="noopener noreferrer">
                      {incident.metadata_path_s3}
                    </a>
                  </span>
                </div>
              )}
              {incident.aws_uploaded !== undefined && (
                <div className="metadata-item">
                  <span className="metadata-label">AWS Uploaded</span>
                  <span className="metadata-value">{incident.aws_uploaded ? 'Yes' : 'No'}</span>
                </div>
              )}
              {incident.aws_integration_error && (
                <div className="metadata-item error">
                  <span className="metadata-label">AWS Error</span>
                  <span className="metadata-value">{incident.aws_integration_error}</span>
                </div>
              )}
            </div>
          )}

          {/* Reason codes */}
          {incident.reason_codes && incident.reason_codes.length > 0 && (
            <div className="metadata-section">
              <h3>Reason Codes</h3>
              <div className="reason-codes">
                {incident.reason_codes.map((code, idx) => (
                  <span key={idx} className="reason-code">{code}</span>
                ))}
              </div>
            </div>
          )}

          {/* Decision info */}
          {incident.decision_id && (
            <div className="metadata-section">
              <h3>Agent Decision</h3>
              <div className="metadata-item">
                <span className="metadata-label">Decision ID</span>
                <span className="metadata-value">{incident.decision_id}</span>
              </div>
              {incident.human_approval_required && (
                <div className="metadata-item warning">
                  <span className="metadata-label">Human Approval Required</span>
                  <span className="metadata-value">Yes</span>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Timeline */}
        {timeline.length > 0 && (
          <div className="timeline-section">
            <h3>Timeline</h3>
            <div className="timeline">
              {timeline.map((event, idx) => (
                <div key={idx} className="timeline-event">
                  <div className="timeline-time">
                    {event.timestamp ? new Date(event.timestamp * 1000).toLocaleString() : 'Unknown time'}
                  </div>
                  <div className="timeline-content">
                    <div className="timeline-event-type">{event.event_type}</div>
                    <div className="timeline-description">{event.description}</div>
                    {event.details && Object.keys(event.details).length > 0 && (
                      <div className="timeline-details">
                        {Object.entries(event.details).map(([k, v]) => (
                          <span key={k} className="detail-tag">{k}: {v}</span>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Evidence */}
        <div className="evidence-section">
          <h3>Evidence</h3>
          <div className="evidence-container">
            <img
              src={evidenceUrl}
              alt="Evidence"
              style={{ maxWidth: '100%', maxHeight: '400px', border: '1px solid #ccc' }}
              onError={(e) => {
                e.target.onerror = null;
                e.target.alt = 'Evidence image not available';
                e.target.style.display = 'none';
              }}
            />
            {incident.evidence_path && !incident.evidence_path_s3 && (
              <p className="evidence-note">
                Evidence stored locally. Configure AWS S3 for remote storage.
              </p>
            )}
          </div>
        </div>

        {/* Actions */}
        <div className="actions">
          <h3>Actions</h3>
          <div className="action-buttons">
            <button
              onClick={() => onApprove(incidentId)}
              disabled={incident.status === 'approved' || incident.status === 'rejected'}
              className="btn-approve"
            >
              Approve
            </button>
            <button
              onClick={() => onReject(incidentId)}
              disabled={incident.status === 'rejected' || incident.status === 'approved'}
              className="btn-reject"
            >
              Reject
            </button>
            {incident.status === 'approved' && (
              <span className="status-confirmed-text">✓ Approved</span>
            )}
            {incident.status === 'rejected' && (
              <span className="status-rejected-text">✗ Rejected</span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default IncidentDetail;