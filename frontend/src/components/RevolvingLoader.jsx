import React from 'react';
import { Sparkles } from 'lucide-react';

const RevolvingLoader = ({
  badge = "Neural Pipeline Active",
  title = "Rendering Diagnostic Analysis...",
  subtitle = "Executing ConvNeXt deep neural layers and synthesizing multi-scale Grad-CAM heatmaps...",
  minHeight = "340px"
}) => {
  return (
    <div className="glass-card rendering-page-view" style={{ minHeight }}>
      {/* Dynamic Revolving Circular Indicator */}
      <div className="revolving-circle-loader">
        <div className="revolving-outer-ring" />
        <div className="revolving-middle-ring" />
        <div className="revolving-inner-core">
          <Sparkles size={14} color="#06b6d4" />
        </div>
      </div>

      {/* Modern Badge */}
      {badge && (
        <div className="rendering-badge">
          <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: 'var(--accent-cyan)' }} />
          <span>{badge}</span>
        </div>
      )}

      {/* Main Title & Subtitle */}
      <h4 className="rendering-title">{title}</h4>
      <p className="rendering-subtitle">{subtitle}</p>

      {/* Shimmering Progress Bar */}
      <div className="rendering-progress-bar">
        <div className="rendering-progress-fill" />
      </div>
    </div>
  );
};

export default RevolvingLoader;
