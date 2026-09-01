import React from 'react';
import { Activity, ShieldCheck, AlertCircle } from 'lucide-react';

const AnalysisResults = ({ result }) => {
  if (!result) {
    return (
      <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '340px', textAlign: 'center' }}>
        <Activity size={48} color="var(--text-muted)" style={{ marginBottom: '16px', opacity: 0.4 }} />
        <h4 style={{ fontSize: '15px', fontWeight: '600', color: 'var(--text-secondary)' }}>Awaiting Scan Input</h4>
        <p style={{ fontSize: '13px', color: 'var(--text-muted)', maxWidth: '280px', marginTop: '4px' }}>
          Upload a CT scan image and click Analyze Image to view diagnosis & Grad-CAM explainability.
        </p>
      </div>
    );
  }

  const prediction = result.image || 'Unknown';
  const confidence = result.confidence || 0.0;
  const probabilities = result.probabilities || {};

  const getClassColor = (cls) => {
    switch (cls.toLowerCase()) {
      case 'normal': return 'var(--accent-emerald)';
      case 'cyst': return 'var(--accent-amber)';
      case 'stone': return 'var(--accent-violet)';
      case 'tumor': return 'var(--accent-rose)';
      default: return 'var(--accent-blue)';
    }
  };

  return (
    <div className="glass-card">
      <div style={{ textAlign: 'center', marginBottom: '24px' }}>
        <div style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: '8px',
          padding: '8px 20px',
          borderRadius: '100px',
          background: `rgba(${prediction.toLowerCase() === 'normal' ? '16, 185, 129' : '244, 63, 94'}, 0.12)`,
          border: `1px solid ${getClassColor(prediction)}`,
          color: getClassColor(prediction),
          fontSize: '22px',
          fontWeight: '800',
          marginBottom: '8px'
        }}>
          {prediction.toLowerCase() === 'normal' ? <ShieldCheck size={24} /> : <AlertCircle size={24} />}
          <span>{prediction}</span>
        </div>
        <div style={{ fontSize: '14px', color: 'var(--text-secondary)' }}>
          Confidence: <strong style={{ color: 'var(--text-primary)' }}>{confidence.toFixed(1)}%</strong>
        </div>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
        <div style={{ fontSize: '12px', fontWeight: '700', textTransform: 'uppercase', letterSpacing: '1px', color: 'var(--text-muted)' }}>
          Class Probabilities
        </div>

        {['Normal', 'Cyst', 'Stone', 'Tumor'].map((cls) => {
          const prob = probabilities[cls] || 0.0;
          const color = getClassColor(cls);
          return (
            <div key={cls}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '13px', fontWeight: '600', marginBottom: '6px' }}>
                <span style={{ color: 'var(--text-primary)' }}>{cls}</span>
                <span style={{ color: color }}>{prob.toFixed(1)}%</span>
              </div>
              <div style={{ height: '8px', width: '100%', background: 'rgba(255, 255, 255, 0.05)', borderRadius: '100px', overflow: 'hidden' }}>
                <div
                  style={{
                    height: '100%',
                    width: `${prob}%`,
                    background: color,
                    borderRadius: '100px',
                    transition: 'width 0.8s ease-out'
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default AnalysisResults;
