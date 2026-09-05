import React, { useEffect, useState } from 'react';
import { Activity, ShieldCheck, AlertCircle, Sparkles, Award, BarChart3 } from 'lucide-react';
import RevolvingLoader from './RevolvingLoader';

const AnalysisResults = ({ result, loading }) => {
  if (loading) {
    return (
      <RevolvingLoader
        badge="Neural Inference Active"
        title="Rendering Diagnostic Analysis..."
        subtitle="Executing ConvNeXt deep neural layers and synthesizing multi-scale Grad-CAM explainability heatmaps..."
      />
    );
  }

  if (!result) {
    return (
      <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '360px', textAlign: 'center' }}>
        <div style={{
          width: '64px',
          height: '64px',
          borderRadius: '50%',
          background: 'rgba(59, 130, 246, 0.1)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          marginBottom: '16px',
          border: '1px dashed rgba(59, 130, 246, 0.3)'
        }}>
          <Activity size={32} color="var(--accent-blue)" style={{ opacity: 0.6 }} />
        </div>
        <h4 style={{ fontSize: '16px', fontWeight: '700', color: 'var(--text-primary)' }}>Awaiting Diagnostic Scan</h4>
        <p style={{ fontSize: '13px', color: 'var(--text-muted)', maxWidth: '280px', marginTop: '6px', lineHeight: '1.5' }}>
          Upload a renal CT scan image and click <strong>Analyze Image</strong> to generate real-time AI classification & Grad-CAM visual explainability.
        </p>
      </div>
    );
  }

  const prediction = result.image || 'Unknown';
  const targetConfidence = result.confidence || 0.0;
  const rawProbabilities = result.probabilities || {};

  // Animated numerical counter state for smooth 60fps transitions
  const [animatedConfidence, setAnimatedConfidence] = useState(0.0);
  const [animatedProbs, setAnimatedProbs] = useState({
    Normal: 0.0,
    Cyst: 0.0,
    Stone: 0.0,
    Tumor: 0.0
  });

  useEffect(() => {
    // Reset start
    setAnimatedConfidence(0.0);
    setAnimatedProbs({ Normal: 0.0, Cyst: 0.0, Stone: 0.0, Tumor: 0.0 });

    const startTime = performance.now();
    const duration = 1200; // 1.2s smooth count-up duration

    const animate = (currentTime) => {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1.0);
      
      // Easing function: easeOutCubic
      const easeProgress = 1 - Math.pow(1 - progress, 3);

      setAnimatedConfidence(targetConfidence * easeProgress);

      setAnimatedProbs({
        Normal: (rawProbabilities.Normal || 0.0) * easeProgress,
        Cyst: (rawProbabilities.Cyst || 0.0) * easeProgress,
        Stone: (rawProbabilities.Stone || 0.0) * easeProgress,
        Tumor: (rawProbabilities.Tumor || 0.0) * easeProgress,
      });

      if (progress < 1.0) {
        requestAnimationFrame(animate);
      }
    };

    const animFrame = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(animFrame);
  }, [result]);

  const getClassColor = (cls) => {
    switch (cls.toLowerCase()) {
      case 'normal': return { main: '#10b981', rgb: '16, 185, 129', label: 'Normal / Healthy' };
      case 'cyst': return { main: '#f59e0b', rgb: '245, 158, 11', label: 'Renal Cyst' };
      case 'stone': return { main: '#8b5cf6', rgb: '139, 92, 246', label: 'Kidney Stone' };
      case 'tumor': return { main: '#f43f5e', rgb: '244, 63, 94', label: 'Renal Tumor' };
      default: return { main: '#3b82f6', rgb: '59, 130, 246', label: cls };
    }
  };

  const predTheme = getClassColor(prediction);

  // SVG Circular Gauge calculation
  const radius = 54;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (animatedConfidence / 100) * circumference;

  // Sorted classes for ranking badges
  const classes = ['Normal', 'Cyst', 'Stone', 'Tumor'];
  const sortedClasses = [...classes].sort((a, b) => (rawProbabilities[b] || 0) - (rawProbabilities[a] || 0));

  return (
    <div className="glass-card" style={{
      animation: 'popIn 0.5s cubic-bezier(0.34, 1.56, 0.64, 1)',
      position: 'relative',
      overflow: 'hidden'
    }}>
      {/* Background Radial Glow */}
      <div style={{
        position: 'absolute',
        top: '-40px',
        right: '-40px',
        width: '180px',
        height: '180px',
        borderRadius: '50%',
        background: `radial-gradient(circle, rgba(${predTheme.rgb}, 0.25) 0%, transparent 70%)`,
        pointerEvents: 'none',
        zIndex: 0
      }} />

      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px', position: 'relative', zIndex: 1 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Sparkles size={18} color="var(--accent-cyan)" />
          <h3 style={{ fontSize: '16px', fontWeight: '800' }}>AI Diagnostic Outcome</h3>
        </div>
        <div style={{
          fontSize: '11px',
          fontWeight: '700',
          padding: '4px 10px',
          borderRadius: '100px',
          background: `rgba(${predTheme.rgb}, 0.15)`,
          color: predTheme.main,
          border: `1px solid rgba(${predTheme.rgb}, 0.3)`,
          display: 'flex',
          alignItems: 'center',
          gap: '4px'
        }}>
          <Award size={13} />
          {targetConfidence >= 90 ? 'High Certainty' : targetConfidence >= 75 ? 'Optimal Confidence' : 'Moderate Confidence'}
        </div>
      </div>

      {/* Radial Gauge & Main Outcome Card */}
      <div style={{
        background: `linear-gradient(135deg, rgba(${predTheme.rgb}, 0.08) 0%, rgba(10, 15, 30, 0.4) 100%)`,
        border: `1px solid rgba(${predTheme.rgb}, 0.25)`,
        borderRadius: '16px',
        padding: '20px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        marginBottom: '24px',
        gap: '16px',
        position: 'relative',
        zIndex: 1,
        boxShadow: `0 0 24px rgba(${predTheme.rgb}, 0.12)`
      }}>
        <div>
          <div style={{ fontSize: '11px', textTransform: 'uppercase', letterSpacing: '1px', fontWeight: '700', color: 'var(--text-muted)', marginBottom: '4px' }}>
            Primary Classification
          </div>
          <div style={{ fontSize: '24px', fontWeight: '900', color: predTheme.main, display: 'flex', alignItems: 'center', gap: '8px' }}>
            {prediction.toLowerCase() === 'normal' ? <ShieldCheck size={28} /> : <AlertCircle size={28} />}
            <span>{prediction}</span>
          </div>
          <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '4px', maxWidth: '190px' }}>
            {prediction.toLowerCase() === 'normal'
              ? 'No renal lesions detected in CT scan.'
              : `High feature alignment detected for ${prediction.toLowerCase()}.`}
          </p>
        </div>

        {/* Dynamic Circular Radial Progress Ring */}
        <div style={{ position: 'relative', width: '120px', height: '120px', flexShrink: 0, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <svg width="120" height="120" viewBox="0 0 120 120" style={{ transform: 'rotate(-90deg)' }}>
            {/* Track Circle */}
            <circle
              cx="60"
              cy="60"
              r={radius}
              stroke="rgba(255, 255, 255, 0.06)"
              strokeWidth="10"
              fill="transparent"
            />
            {/* Dynamic Fill Circle */}
            <circle
              cx="60"
              cy="60"
              r={radius}
              stroke={predTheme.main}
              strokeWidth="10"
              fill="transparent"
              strokeDasharray={circumference}
              strokeDashoffset={strokeDashoffset}
              strokeLinecap="round"
              style={{
                transition: 'stroke-dashoffset 0.1s linear',
                filter: `drop-shadow(0 0 6px ${predTheme.main})`
              }}
            />
          </svg>
          <div style={{ position: 'absolute', textAlign: 'center', inset: 0, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
            <span style={{ fontSize: '20px', fontWeight: '900', color: 'var(--text-primary)', lineHeight: 1 }}>
              {animatedConfidence.toFixed(1)}%
            </span>
            <span style={{ fontSize: '10px', fontWeight: '600', color: 'var(--text-muted)', marginTop: '2px' }}>
              Confidence
            </span>
          </div>
        </div>
      </div>

      {/* Class Probability Distribution Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px', position: 'relative', zIndex: 1 }}>
        <div style={{ fontSize: '13px', fontWeight: '800', textTransform: 'uppercase', letterSpacing: '0.8px', color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '6px' }}>
          <BarChart3 size={15} color="var(--accent-blue)" />
          Multi-Class Probability Breakdown
        </div>
      </div>

      {/* Animated Class Progress Bars */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', position: 'relative', zIndex: 1 }}>
        {classes.map((cls) => {
          const rawVal = rawProbabilities[cls] || 0.0;
          const animVal = animatedProbs[cls] || 0.0;
          const theme = getClassColor(cls);
          const isTop = cls === prediction;
          const rankIndex = sortedClasses.indexOf(cls) + 1;

          return (
            <div
              key={cls}
              style={{
                background: isTop ? `rgba(${theme.rgb}, 0.08)` : 'rgba(255, 255, 255, 0.02)',
                border: `1px solid ${isTop ? `rgba(${theme.rgb}, 0.4)` : 'var(--border-glass)'}`,
                borderRadius: '12px',
                padding: '12px 14px',
                transition: 'all 0.3s ease',
                boxShadow: isTop ? `0 0 16px rgba(${theme.rgb}, 0.15)` : 'none'
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span style={{
                    width: '8px',
                    height: '8px',
                    borderRadius: '50%',
                    backgroundColor: theme.main,
                    boxShadow: `0 0 8px ${theme.main}`
                  }} />
                  <span style={{ fontSize: '13px', fontWeight: isTop ? '800' : '600', color: isTop ? 'var(--text-primary)' : 'var(--text-secondary)' }}>
                    {cls}
                  </span>
                  {isTop && (
                    <span style={{
                      fontSize: '10px',
                      fontWeight: '800',
                      padding: '2px 8px',
                      borderRadius: '100px',
                      background: theme.main,
                      color: '#000'
                    }}>
                      TOP MATCH
                    </span>
                  )}
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span style={{ fontSize: '11px', color: 'var(--text-muted)', fontWeight: '600' }}>#{rankIndex}</span>
                  <span style={{ fontSize: '14px', fontWeight: '800', color: theme.main, minWidth: '48px', textAlign: 'right' }}>
                    {animVal.toFixed(1)}%
                  </span>
                </div>
              </div>

              {/* Dynamic Animated Progress Bar */}
              <div style={{
                height: '10px',
                width: '100%',
                background: 'rgba(0, 0, 0, 0.3)',
                borderRadius: '100px',
                overflow: 'hidden',
                position: 'relative'
              }}>
                <div
                  style={{
                    height: '100%',
                    width: `${animVal}%`,
                    background: `linear-gradient(90deg, ${theme.main} 0%, rgba(${theme.rgb}, 0.7) 100%)`,
                    borderRadius: '100px',
                    boxShadow: `0 0 12px ${theme.main}`,
                    transition: 'width 0.1s linear',
                    position: 'relative'
                  }}
                >
                  {/* Subtle animated light highlight on the bar */}
                  <div style={{
                    position: 'absolute',
                    top: 0,
                    right: 0,
                    bottom: 0,
                    width: '20px',
                    background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.6), transparent)',
                    borderRadius: '100px'
                  }} />
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default AnalysisResults;
