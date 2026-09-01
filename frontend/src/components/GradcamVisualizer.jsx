import React, { useState } from 'react';
import { Flame, Eye, Layers, Image as ImageIcon } from 'lucide-react';

const GradcamVisualizer = ({ result, originalImage }) => {
  const [activeTab, setActiveTab] = useState('sidebyside');

  if (!result || !result.gradcam) return null;

  const heatmapClass = result.heatmap_class || result.image || '';
  const isNormalHeatmap = heatmapClass.toLowerCase() === 'normal';

  return (
    <div className="gradcam-section">
      <div className="gradcam-header">
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '15px', fontWeight: '700' }}>
            <Flame size={18} color="var(--accent-amber)" />
            <span>Grad-CAM Explainability Map</span>
          </div>
          <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '2px' }}>
            {isNormalHeatmap
              ? '✅ Healthy CT Scan: Zero pathological regions or lesions detected.'
              : `🔥 Visualizing regions influencing ${heatmapClass} features.`}
          </div>
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
            <span className="gradcam-img-label">Attention Overlay</span>
          </div>
        )}

        {activeTab === 'sidebyside' && (
          <div style={{ display: 'flex', gap: '16px', width: '100%', justifyContent: 'center' }}>
            <div className="gradcam-img-card">
              <img src={originalImage} alt="Original Scan" />
              <span className="gradcam-img-label">Original Scan</span>
            </div>
            <div className="gradcam-img-card">
              <img src={result.gradcam} alt="Grad-CAM Overlay" />
              <span className="gradcam-img-label">Attention Overlay</span>
            </div>
          </div>
        )}

        {activeTab === 'heatmap' && (
          <div className="gradcam-img-card">
            <img src={result.heatmap || result.gradcam} alt="Raw Heatmap" />
            <span className="gradcam-img-label">Standalone Heatmap</span>
          </div>
        )}
      </div>

      <div className="gradcam-legend">
        <span>Low Influence</span>
        <div className="gradcam-bar" />
        <span>High Influence</span>
      </div>
    </div>
  );
};

export default GradcamVisualizer;
