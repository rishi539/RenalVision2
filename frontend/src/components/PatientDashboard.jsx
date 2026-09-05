import React, { useState } from 'react';
import { 
  Activity, 
  TrendingUp, 
  Calendar, 
  CheckCircle2, 
  AlertTriangle, 
  Trash2, 
  Stethoscope, 
  Utensils, 
  Clock, 
  Sparkles, 
  HeartPulse, 
  Check, 
  Info, 
  WifiOff, 
  RefreshCw,
  FileText
} from 'lucide-react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, Legend } from 'recharts';
import RevolvingLoader from './RevolvingLoader';
import { generatePdfReport } from '../utils/generatePdfReport';

const PatientDashboard = ({ user, historyData, loading, error, onRetry, onNavigateToRecs }) => {
  if (!user) return null;

  // 1. Initial Timeline Loading State with Revolving Circle
  if (loading) {
    return (
      <RevolvingLoader
        badge="Patient Portal Sync"
        title="Loading Health History..."
        subtitle="Retrieving sequential CT scans, longitudinal trajectory, and personal health telemetry..."
      />
    );
  }

  // 2. Disconnected / Error State
  if (error) {
    return (
      <div className="glass-card" style={{
        padding: '40px 24px',
        textAlign: 'center',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: '340px',
        background: 'rgba(244, 63, 94, 0.05)',
        border: '1px solid rgba(244, 63, 94, 0.25)',
        borderRadius: '16px',
        animation: 'popIn 0.4s ease-out'
      }}>
        <div style={{
          width: '56px',
          height: '56px',
          borderRadius: '50%',
          background: 'rgba(244, 63, 94, 0.12)',
          border: '1px solid rgba(244, 63, 94, 0.3)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          marginBottom: '16px',
          boxShadow: '0 0 20px rgba(244, 63, 94, 0.2)'
        }}>
          <WifiOff size={28} color="#f43f5e" />
        </div>

        <div style={{
          fontSize: '11px',
          fontWeight: '800',
          textTransform: 'uppercase',
          letterSpacing: '1px',
          padding: '3px 10px',
          borderRadius: '100px',
          background: 'rgba(244, 63, 94, 0.15)',
          color: '#f43f5e',
          marginBottom: '10px'
        }}>
          Backend Service Offline
        </div>

        <h4 style={{ fontSize: '17px', fontWeight: '800', color: 'var(--text-primary)', marginBottom: '6px' }}>
          Unable to Connect to Recommendation Engine
        </h4>
        <p style={{ fontSize: '13px', color: 'var(--text-secondary)', maxWidth: '380px', lineHeight: '1.5', marginBottom: '20px' }}>
          Could not establish connection with backend server at <code style={{ background: 'rgba(0,0,0,0.4)', padding: '2px 6px', borderRadius: '4px', color: 'var(--accent-cyan)' }}>http://localhost:8080</code>.
        </p>

        {onRetry && (
          <button
            onClick={onRetry}
            className="predict-btn"
            style={{ width: 'auto', padding: '10px 24px', fontSize: '13px' }}
          >
            <RefreshCw size={15} />
            <span>Retry Connection</span>
          </button>
        )}
      </div>
    );
  }

  const scans = historyData?.scans || [];

  const handleGenerateReportForScan = (targetScan) => {
    if (!targetScan || !user) return;
    generatePdfReport({
      user,
      scan: targetScan,
      originalImage: targetScan.original_b64 || targetScan.gradcam_b64,
      gradcamImage: targetScan.gradcam_b64,
      recommendations: historyData?.recommendations,
      trendStatus: historyData?.trend_status
    });
  };

  const handleGenerateLatestReport = () => {
    if (scans.length === 0) {
      alert('Please upload and analyze a CT scan first to generate your clinical diagnostic report.');
      return;
    }
    const latestScan = scans[scans.length - 1];
    handleGenerateReportForScan(latestScan);
  };

  const chartData = scans.map((scan) => ({
    scan: `Scan #${scan.scan_number}`,
    Normal: scan.prob_normal,
    Cyst: scan.prob_cyst,
    Stone: scan.prob_stone,
    Tumor: scan.prob_tumor,
  }));


  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      
      {/* 1. Main Graph & Patient Header Card */}
      <div className="glass-card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px', flexWrap: 'wrap', gap: '12px' }}>
          <div>
            <h3 style={{ fontSize: '18px', fontWeight: '800', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <HeartPulse style={{ color: 'var(--accent-blue)' }} size={20} />
              My Personal Health & Recovery Dashboard
            </h3>
            <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '2px' }}>
              Sequential CT scan history for <strong style={{ color: 'var(--text-primary)' }}>{user.name}</strong> (@{user.username})
            </p>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap' }}>
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              padding: '5px 12px',
              borderRadius: '100px',
              fontSize: '12px',
              fontWeight: '700',
              background: historyData?.trend_color === 'emerald' ? 'rgba(16, 185, 129, 0.12)' : 'rgba(245, 158, 11, 0.12)',
              color: historyData?.trend_color === 'emerald' ? 'var(--accent-emerald)' : 'var(--accent-amber)',
              border: `1px solid ${historyData?.trend_color === 'emerald' ? 'rgba(16, 185, 129, 0.3)' : 'rgba(245, 158, 11, 0.3)'}`
            }}>
              {historyData?.trend_color === 'emerald' ? <CheckCircle2 size={15} /> : <AlertTriangle size={15} />}
              <span>{historyData?.trend_status || 'No Scans Logged Yet'}</span>
            </div>

            <button
              onClick={handleGenerateLatestReport}
              className="report-download-btn"
              disabled={scans.length === 0}
              style={scans.length === 0 ? { opacity: 0.5, cursor: 'not-allowed' } : {}}
              title={scans.length > 0 ? "Download Hospital-Grade Clinical Diagnostic PDF Report" : "Upload a scan first to generate a report"}
            >
              <FileText size={15} />
              <span>Generate Report</span>
            </button>
          </div>
        </div>

        {/* Recharts Progression Line Graph */}
        {chartData.length > 0 ? (
          <div style={{ height: '280px', width: '100%', marginBottom: '20px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                <XAxis dataKey="scan" stroke="var(--text-muted)" fontSize={12} />
                <YAxis domain={[0, 100]} stroke="var(--text-muted)" fontSize={12} unit="%" />
                <Tooltip
                  contentStyle={{ backgroundColor: 'var(--bg-secondary)', borderColor: 'var(--border-glass)', borderRadius: '10px', fontSize: '12px' }}
                />
                <Legend wrapperStyle={{ paddingTop: '10px', fontSize: '12px' }} />
                <Line type="monotone" dataKey="Normal" stroke="#10b981" strokeWidth={3} dot={{ r: 5 }} />
                <Line type="monotone" dataKey="Cyst" stroke="#f59e0b" strokeWidth={2} dot={{ r: 4 }} />
                <Line type="monotone" dataKey="Stone" stroke="#8b5cf6" strokeWidth={2} dot={{ r: 4 }} />
                <Line type="monotone" dataKey="Tumor" stroke="#f43f5e" strokeWidth={2} dot={{ r: 4 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <div style={{ textAlign: 'center', padding: '30px 0', color: 'var(--text-muted)', fontSize: '13px' }}>
            No CT scan history logged yet. Upload a CT scan image above to generate your first health record & progression graph!
          </div>
        )}
      </div>

      {/* 2. Referral to new AI Recommendations Tab */}
      <div className="glass-card" style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '20px 24px',
        flexWrap: 'wrap',
        gap: '14px',
        background: 'linear-gradient(135deg, rgba(15, 23, 42, 0.6) 0%, rgba(6, 182, 212, 0.06) 100%)',
        border: '1px solid rgba(6, 182, 212, 0.2)'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{
            width: '40px',
            height: '40px',
            borderRadius: '10px',
            background: 'rgba(6, 182, 212, 0.15)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}>
            <Sparkles size={20} color="var(--accent-cyan)" />
          </div>
          <div>
            <h4 style={{ fontSize: '15px', fontWeight: '800', color: 'var(--text-primary)', margin: 0 }}>
              AI Clinical Recommendations
            </h4>
            <p style={{ fontSize: '12px', color: 'var(--text-secondary)', margin: '2px 0 0 0' }}>
              Personalized care intelligence is now available in the dedicated <strong>AI Recommendations</strong> tab on the left sidebar.
            </p>
          </div>
        </div>
        {onNavigateToRecs && (
          <button
            onClick={onNavigateToRecs}
            className="ai-analysis-btn"
            style={{ padding: '8px 18px', fontSize: '12.5px' }}
          >
            <Sparkles size={14} />
            <span>Open AI Recommendations</span>
          </button>
        )}
      </div>

      {/* 3. Scan History Cards */}
      {historyData?.scans && historyData.scans.length > 0 && (
        <div className="glass-card">
          <h4 style={{ fontSize: '14px', fontWeight: '800', marginBottom: '14px', color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Calendar size={16} style={{ color: 'var(--accent-cyan)' }} />
            My Scan History Timeline ({historyData.scans.length})
          </h4>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: '12px' }}>
            {[...historyData.scans].reverse().map((s) => (
              <div key={s.id} style={{ background: 'rgba(0,0,0,0.3)', border: '1px solid var(--border-glass)', borderRadius: 'var(--radius-sm)', padding: '12px', position: 'relative' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', fontWeight: '700', marginBottom: '6px' }}>
                  <span>Scan #{s.scan_number}</span>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{ color: 'var(--text-muted)' }}>{s.scan_date.split(' ')[0]}</span>
                    <button 
                      onClick={() => handleGenerateReportForScan(s)}
                      style={{ background: 'none', border: 'none', color: 'var(--accent-cyan)', cursor: 'pointer', padding: 0, display: 'flex', alignItems: 'center' }}
                      title={`Download Clinical Diagnostic PDF for Scan #${s.scan_number}`}
                    >
                      <FileText size={14} />
                    </button>
                    <button 
                      onClick={() => window.handleDeleteScan && window.handleDeleteScan(s.id)}
                      style={{ background: 'none', border: 'none', color: 'var(--accent-rose)', cursor: 'pointer', padding: 0 }}
                      title="Delete Scan"
                    >
                      <Trash2 size={14} />
                    </button>
                  </div>
                </div>
                <div style={{ fontSize: '13px', fontWeight: '700', color: s.prediction === 'Normal' ? 'var(--accent-emerald)' : 'var(--accent-rose)' }}>
                  {s.prediction} ({s.confidence.toFixed(1)}%)
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

    </div>
  );
};

export default PatientDashboard;
