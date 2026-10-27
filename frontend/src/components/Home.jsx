import React, { useState, useEffect } from 'react';
import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

const Home = () => {
  const [video, setVideo] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [uploadProgress, setUploadProgress] = useState(0);

  const events = Array.isArray(analysisResult?.events)
    ? analysisResult.events
    : [];

  const decisions = Array.isArray(analysisResult?.decisions)
    ? analysisResult.decisions
    : [];

  const inferenceMetrics = analysisResult?.inference_metrics || null;
  const videoMetadata = analysisResult?.video_metadata || null;

  const handleAnalyze = async (e) => {
    e.preventDefault();

    if (!video) {
      setError('Please select a video file first.');
      return;
    }

    setLoading(true);
    setError('');
    setAnalysisResult(null);
    setUploadProgress(0);

    const formData = new FormData();
    formData.append('file', video);

    try {
      // Use a generous timeout for video processing (5 minutes)
      // This prevents the browser from hanging indefinitely on large videos
      const ANALYSIS_TIMEOUT_MS = 5 * 60 * 1000;

      const response = await axios.post(
        `${API_URL}/analyze`,
        formData,
        {
          timeout: ANALYSIS_TIMEOUT_MS,
          onUploadProgress: (progressEvent) => {
            const progress = Math.round((progressEvent.loaded * 100) / (progressEvent.total || 1));
            setUploadProgress(progress);
          }
        }
      );

      console.log('Analysis API response:', response.data);

      if (!response.data || typeof response.data !== 'object') {
        throw new Error('The server returned an invalid response.');
      }

      setAnalysisResult(response.data);
    } catch (err) {
      console.error('Video analysis failed:', err);

      if (err.code === 'ECONNABORTED' || err.message?.includes('timeout')) {
        setError(
          `Request timed out after ${ANALYSIS_TIMEOUT_MS / 1000 / 60} minutes. The video may be too large or the backend is overloaded.`
        );
      } else if (err.response) {
        const detail = err.response.data?.detail;
        setError(
          `Server error (${err.response.status}): ${
            typeof detail === 'string' ? detail : err.response.statusText
          }`
        );
      } else if (err.request) {
        setError(
          'No response from the backend. Check that the API is running at ' + API_URL + '.'
        );
      } else {
        setError(err.message || 'An unexpected error occurred.');
      }
    } finally {
      setLoading(false);
      setUploadProgress(0);
    }
  };

  const getSeverityClass = (severity) => {
    switch (severity) {
      case 'critical': return 'severity-critical';
      case 'high': return 'severity-high';
      case 'medium': return 'severity-medium';
      case 'low': return 'severity-low';
      default: return '';
    }
  };

  const formatDecision = (dec) => {
    const parts = [];
    if (dec.risk_level) parts.push(`Risk: ${dec.risk_level.toUpperCase()}`);
    if (dec.reason_codes && dec.reason_codes.length) parts.push(`Codes: ${dec.reason_codes.join(', ')}`);
    if (dec.verification_result) parts.push(`Verification: ${dec.verification_result}`);
    if (dec.human_approval_required) parts.push('⚠ Human Approval Required');
    return parts.join(' | ');
  };

  return (
    <main className="home">
      <div className="header-bar">
        <h2>Video Analysis</h2>
        <BackendStatus />
      </div>

      <form onSubmit={handleAnalyze} className="video-upload-form">
        <div>
          <label htmlFor="video">Select Video File:</label>
          <input
            type="file"
            id="video"
            accept="video/*"
            onChange={(e) => {
              setVideo(e.target.files?.[0] || null);
              setError('');
              setAnalysisResult(null);
            }}
            disabled={loading}
          />
          {video && <p>Selected: {video.name} ({(video.size / 1024 / 1024).toFixed(2)} MB)</p>}
        </div>

        <button type="submit" disabled={loading || !video}>
          {loading ? 'Analyzing video...' : 'Analyze Video'}
        </button>
      </form>

      {loading && (
        <div className="progress-container">
          <div className="progress-bar">
            <div
              className="progress-fill"
              style={{ width: `${uploadProgress}%` }}
            ></div>
          </div>
          <p className="progress-text">Uploading... {uploadProgress}%</p>
        </div>
      )}

      {error && (
        <div className="error" role="alert">
          <strong>Analysis failed:</strong> {error}
        </div>
      )}

      {analysisResult && (
        <section className="results" aria-live="polite">
          <div className="results-header">
            <h3>Analysis Results</h3>
            {analysisResult.incident_id && (
              <span className="incident-badge">
                Incident: <code>{analysisResult.incident_id.slice(0, 8)}...</code>
              </span>
            )}
          </div>

          <p>
            <strong>Message:</strong>{' '}
            {analysisResult.message || 'Analysis completed.'}
          </p>

          {/* Video Metadata */}
          {videoMetadata && (
            <div className="metadata-section">
              <h4>Video Metadata</h4>
              <div className="metadata-grid">
                <div className="metadata-item">
                  <span className="metadata-label">Resolution</span>
                  <span className="metadata-value">{videoMetadata.width}×{videoMetadata.height}</span>
                </div>
                <div className="metadata-item">
                  <span className="metadata-label">FPS</span>
                  <span className="metadata-value">{videoMetadata.fps.toFixed(1)}</span>
                </div>
                <div className="metadata-item">
                  <span className="metadata-label">Frames</span>
                  <span className="metadata-value">{videoMetadata.frame_count}</span>
                </div>
                <div className="metadata-item">
                  <span className="metadata-label">Duration</span>
                  <span className="metadata-value">{videoMetadata.duration.toFixed(1)}s</span>
                </div>
              </div>
            </div>
          )}

          {/* Inference Metrics */}
          {inferenceMetrics && (
            <div className="metrics-section">
              <h4>Inference Metrics</h4>
              <div className="metrics-grid">
                <div className="metric-card">
                  <span className="metric-value">{inferenceMetrics.total_frames}</span>
                  <span className="metric-label">Frames Processed</span>
                </div>
                <div className="metric-card">
                  <span className="metric-value">{inferenceMetrics.total_detections}</span>
                  <span className="metric-label">Total Detections</span>
                </div>
                <div className="metric-card">
                  <span className="metric-value">{inferenceMetrics.total_events}</span>
                  <span className="metric-label">Events Generated</span>
                </div>
                <div className="metric-card">
                  <span className="metric-value">{inferenceMetrics.avg_fps?.toFixed(1) || 'N/A'}</span>
                  <span className="metric-label">Avg FPS</span>
                </div>
                <div className="metric-card">
                  <span className="metric-value">{(inferenceMetrics.avg_frame_time * 1000)?.toFixed(1) || 'N/A'} ms</span>
                  <span className="metric-label">Avg Frame Time</span>
                </div>
                <div className="metric-card">
                  <span className="metric-value">{(inferenceMetrics.total_processing_time)?.toFixed(1) || 'N/A'}s</span>
                  <span className="metric-label">Total Time</span>
                </div>
              </div>
            </div>
          )}

          {/* Events */}
          <h4>Events ({events.length})</h4>

          {events.length === 0 ? (
            <p className="no-events">No safety events were detected in this video.</p>
          ) : (
            <div className="event-list">
              {events.map((ev, idx) => (
                <article
                  key={`${ev.person_id}-${ev.event_type}-${idx}`}
                  className={`event-item ${ev.status ? `status-${ev.status}` : ''}`}
                >
                  <div className="event-header">
                    <strong>{String(ev.event_type || 'Unknown event').replaceAll('_', ' ')}</strong>
                    {ev.severity && (
                      <span className={`severity-badge ${getSeverityClass(ev.severity)}`}>
                        {ev.severity.toUpperCase()}
                      </span>
                    )}
                  </div>
                  <div className="event-details">
                    <p>Person: {ev.person_id ?? 'Unknown'}</p>
                    <p>Zone: {ev.zone || 'Not specified'}</p>
                    <p>Status: {ev.status || 'Unknown'}</p>
                    <p>
                      Confidence:{' '}
                      {Number.isFinite(ev.confidence)
                        ? `${(ev.confidence * 100).toFixed(1)}%`
                        : 'N/A'}
                    </p>
                    <p>
                      Duration:{' '}
                      {Number.isFinite(ev.duration)
                        ? `${ev.duration.toFixed(2)} seconds`
                        : 'N/A'}
                    </p>
                    <p>
                      Time:{' '}
                      {Number.isFinite(ev.timestamp)
                        ? new Date(ev.timestamp * 1000).toLocaleString()
                        : 'N/A'}
                      </p>
                  </div>
                </article>
              ))}
            </div>
          )}

          {/* Agent Decisions */}
          <h4>Agent Decisions ({decisions.length})</h4>

          {decisions.length === 0 ? (
            <p>No agent decisions were returned.</p>
          ) : (
            <div className="decision-list">
              {decisions.map((dec, idx) => (
                <article
                  key={`${dec.person_id}-${dec.decision}-${idx}`}
                  className={`decision-item ${dec.risk_level ? `risk-${dec.risk_level}` : ''}`}
                >
                  <div className="decision-header">
                    <strong>{dec.decision || 'Unknown decision'}</strong>
                    {dec.risk_level && (
                      <span className={`risk-badge risk-${dec.risk_level}`}>
                        {dec.risk_level.toUpperCase()}
                      </span>
                    )}
                  </div>
                  <div className="decision-details">
                    <p>Person: {dec.person_id ?? 'Unknown'}</p>
                    <p>{dec.reason || 'No reason provided.'}</p>
                    <p><strong>Next action:</strong> {dec.next_action || 'None'}</p>
                    {formatDecision(dec) && (
                      <p className="decision-meta">{formatDecision(dec)}</p>
                    )}
                  </div>
                </article>
              ))}
            </div>
          )}
        </section>
      )}
    </main>
  );
};

// Backend Status Component
const BackendStatus = () => {
  const [status, setStatus] = useState('checking');

  useEffect(() => {
    const checkStatus = async () => {
      try {
        const response = await axios.get(`${API_URL}/health`, { timeout: 3000 });
        if (response.data.status === 'ok') {
          setStatus('online');
        } else {
          setStatus('degraded');
        }
      } catch {
        setStatus('offline');
      }
    };
    checkStatus();
    const interval = setInterval(checkStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  const statusColors = {
    online: 'status-online',
    degraded: 'status-degraded',
    offline: 'status-offline',
    checking: 'status-checking'
  };

  const statusLabels = {
    online: 'Backend Online',
    degraded: 'Backend Degraded',
    offline: 'Backend Offline',
    checking: 'Checking...'
  };

  return (
    <div className={`backend-status ${statusColors[status]}`}>
      <span className="status-indicator"></span>
      <span>{statusLabels[status]}</span>
    </div>
  );
};

export default Home;