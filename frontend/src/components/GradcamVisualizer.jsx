import React, { useState } from 'react';
import { Flame, Eye, Layers, Image as ImageIcon, CheckCircle2, AlertCircle, Cpu } from 'lucide-react';

const GradcamVisualizer = ({ result, originalImage }) => {
  const [activeTab, setActiveTab] = useState('sidebyside');

  if (!result || !result.gradcam) return null;

  const heatmapClass = result.heatmap_class || result.image || '';
  const isHighlighted = result.is_highlighted !== false;
  const normalProb = result.probabilities?.Normal || 0;

  return (
    <div className="gradcam-section">
      <div className="gradcam-header">
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '15px', fontWeight: '700' }}>
            {isHighlighted ? (
              <Flame size={18} color="var(--accent-rose)" />
            ) : (
              <CheckCircle2 size={18} color="var(--accent-emerald)" />
            )}
            <span>Grad-CAM Explainability Map</span>
            <span style={{
              fontSize: '11px',
              padding: '2px 8px',
              borderRadius: '100px',
              fontWeight: '700',
              background: isHighlighted ? 'rgba(244, 63, 94, 0.15)' : 'rgba(16, 185, 129, 0.15)',
              color: isHighlighted ? 'var(--accent-rose)' : 'var(--accent-emerald)',
              border: `1px solid ${isHighlighted ? 'rgba(244, 63, 94, 0.3)' : 'rgba(16, 185, 129, 0.3)'}`
            }}>
              {isHighlighted ? `Active (${heatmapClass})` : 'Suppressed (Normal)'}
            </span>
          </div>
          <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '4px' }}>
            {isHighlighted
              ? `🔥 Pinpointing regions influencing ${heatmapClass} features (${(result.probabilities?.[heatmapClass] || result.confidence).toFixed(1)}% probability detected).`
              : `✅ Healthy / Normal Scan (${normalProb.toFixed(1)}% Normal): Zero lesions or abnormalities detected. Grad-CAM highlighting is suppressed.`}
          </div>
          {result.gradcam_model && (
            <div style={{
              fontSize: '10px',
              fontWeight: '600',
              color: 'var(--accent-cyan)',
              marginTop: '4px',
              display: 'flex',
              alignItems: 'center',
              gap: '4px'
            }}>
              <Cpu size={11} />
              <span>Heatmap Source: <strong>{result.gradcam_model}</strong></span>
            </div>
          )}
        </div>

        <div className="gradcam-tabs">
          <button
            className={`gradcam-tab ${activeTab === 'overlay' ? 'active' : ''}`}
            onClick={() => setActiveTab('overlay')}
          >
            Overlay
          </button>
          <button
            className={`gradcam-tab ${activeTab === 'sidebyside' ? 'active' : ''}`}
            onClick={() => setActiveTab('sidebyside')}
          >
            Side-by-Side
          </button>
          <button
            className={`gradcam-tab ${activeTab === 'heatmap' ? 'active' : ''}`}
            onClick={() => setActiveTab('heatmap')}
          >
            Heatmap
          </button>
        </div>
      </div>

      <div className="gradcam-display">
        {activeTab === 'overlay' && (
          <div className="gradcam-img-card">
            <img src={result.gradcam} alt="Grad-CAM Overlay" />
            <span className="gradcam-img-label">
              {isHighlighted ? `Attention Overlay (${heatmapClass})` : 'Clean Scan (Highlighting Suppressed)'}
            </span>
          </div>
        )}

        {activeTab === 'sidebyside' && (
          <div style={{ display: 'flex', gap: '16px', width: '100%', justifyContent: 'center', flexWrap: 'wrap' }}>
            <div className="gradcam-img-card">
              <img src={originalImage} alt="Original Scan" />
              <span className="gradcam-img-label">Original CT Scan</span>
            </div>
            <div className="gradcam-img-card">
              <img src={result.gradcam} alt="Grad-CAM Overlay" />
              <span className="gradcam-img-label">
                {isHighlighted ? `Attention Overlay (${heatmapClass})` : 'Clean Scan (Highlighting Suppressed)'}
              </span>
            </div>
          </div>
        )}

        {activeTab === 'heatmap' && (
          isHighlighted && result.heatmap ? (
            <div className="gradcam-img-card">
              <img src={result.heatmap} alt="Raw Heatmap" />
              <span className="gradcam-img-label">{heatmapClass} Activation Heatmap</span>
            </div>
          ) : (
            <div className="gradcam-img-card" style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              minHeight: '260px',
              padding: '30px 20px',
              textAlign: 'center',
              background: 'rgba(16, 185, 129, 0.04)',
              border: '1px dashed rgba(16, 185, 129, 0.25)',
              borderRadius: '12px'
            }}>
              <div style={{
                width: '46px',
                height: '46px',
                borderRadius: '50%',
                background: 'rgba(16, 185, 129, 0.12)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                marginBottom: '12px'
              }}>
                <CheckCircle2 size={24} color="#10b981" />
              </div>
              <h4 style={{ fontSize: '15px', fontWeight: '700', color: 'var(--text-primary)', marginBottom: '6px' }}>
                Heatmap Highlighting Deactivated
              </h4>
              <p style={{ fontSize: '12.5px', color: 'var(--text-secondary)', maxWidth: '380px', lineHeight: '1.5', margin: 0 }}>
                This CT scan is classified as Normal / Healthy ({normalProb.toFixed(1)}%). No pathological lesions (Stone, Tumor, or Cyst) exceed the 10% clinical threshold, so false-positive heatmap coloring is suppressed.
              </p>
            </div>
          )
        )}
      </div>

      <div className="gradcam-legend">
        {isHighlighted ? (
          <>
            <span>Low Influence</span>
            <div className="gradcam-bar" />
            <span>High Influence ({heatmapClass})</span>
          </>
        ) : (
          <span style={{ fontSize: '12px', color: 'var(--accent-emerald)', fontWeight: '600' }}>
            ✔ Heatmap highlighting inactive for healthy tissue (Normal ≥ 80%)
          </span>
        )}
      </div>
    </div>
  );
};

export default GradcamVisualizer;
