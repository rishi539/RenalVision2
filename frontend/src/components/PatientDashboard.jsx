import React from 'react';
import { Activity, TrendingUp, Calendar, CheckCircle2, AlertTriangle, Trash2 } from 'lucide-react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, Legend } from 'recharts';

const PatientDashboard = ({ user, historyData }) => {
  if (!user) return null;

  const scans = historyData?.scans || [];
  const chartData = scans.map((scan) => ({
    scan: `Scan #${scan.scan_number}`,
    Normal: scan.prob_normal,
    Cyst: scan.prob_cyst,
    Stone: scan.prob_stone,
    Tumor: scan.prob_tumor,
  }));

  return (
    <div className="glass-card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px', flexWrap: 'wrap', gap: '12px' }}>
        <div>
          <h3 style={{ fontSize: '18px', fontWeight: '800' }}>
            My Personal Health & Recovery Dashboard
          </h3>
          <p style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
            Sequential CT scan history for <strong style={{ color: 'var(--text-primary)' }}>{user.name}</strong> (@{user.username})
          </p>
        </div>

        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          padding: '6px 14px',
          borderRadius: '100px',
          fontSize: '12px',
          fontWeight: '700',
          background: historyData?.trend_color === 'emerald' ? 'rgba(16, 185, 129, 0.12)' : 'rgba(245, 158, 11, 0.12)',
          color: historyData?.trend_color === 'emerald' ? 'var(--accent-emerald)' : 'var(--accent-amber)',
          border: `1px solid ${historyData?.trend_color === 'emerald' ? 'rgba(16, 185, 129, 0.3)' : 'rgba(245, 158, 11, 0.3)'}`
        }}>
          {historyData?.trend_color === 'emerald' ? <CheckCircle2 size={16} /> : <AlertTriangle size={16} />}
          <span>{historyData?.trend_status || 'No Scans Logged Yet'}</span>
        </div>
      </div>

      {/* Recharts Progression Line Graph */}
      {chartData.length > 0 ? (
        <div style={{ height: '280px', width: '100%', marginBottom: '28px' }}>
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
        <div style={{ textAlign: 'center', padding: '36px 0', color: 'var(--text-muted)', fontSize: '13px' }}>
          No CT scan history logged yet. Upload a CT scan image above to generate your first health record & progression graph!
        </div>
      )}

      {/* Scan History Cards */}
      {historyData?.scans && historyData.scans.length > 0 && (
        <div>
          <h4 style={{ fontSize: '14px', fontWeight: '700', marginBottom: '14px', color: 'var(--text-secondary)' }}>
            My Scan History Timeline ({historyData.scans.length})
          </h4>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(240px, 1fr))', gap: '14px' }}>
            {[...historyData.scans].reverse().map((s) => (
              <div key={s.id} style={{ background: 'rgba(0,0,0,0.3)', border: '1px solid var(--border-glass)', borderRadius: 'var(--radius-sm)', padding: '14px', position: 'relative' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', fontWeight: '700', marginBottom: '6px' }}>
                  <span>Scan #{s.scan_number}</span>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{ color: 'var(--text-muted)' }}>{s.scan_date.split(' ')[0]}</span>
                    <button 
                      onClick={() => window.handleDeleteScan && window.handleDeleteScan(s.id)}
                      style={{ background: 'none', border: 'none', color: 'var(--accent-rose)', cursor: 'pointer', padding: 0 }}
                      title="Delete Scan"
                    >
                      <Trash2 size={14} />
                    </button>
                  </div>
                </div>
                <div style={{ fontSize: '14px', fontWeight: '700', color: s.prediction === 'Normal' ? 'var(--accent-emerald)' : 'var(--accent-rose)' }}>
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
