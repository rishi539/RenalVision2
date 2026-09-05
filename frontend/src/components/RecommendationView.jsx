import React, { useState, useEffect } from 'react';
import { 
  Sparkles, 
  Stethoscope, 
  Utensils, 
  Clock, 
  Check, 
  Info, 
  AlertTriangle, 
  RefreshCw,
  Award,
  HeartPulse
} from 'lucide-react';
import RevolvingLoader from './RevolvingLoader';
import { triggerGeminiRecommendation } from '../services/api';

const RecommendationView = ({ user, historyData, onRefreshHistory, runTrigger, onNavigateToAnalysis }) => {
  const [recs, setRecs] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeCategory, setActiveCategory] = useState('clinical'); // 'clinical' | 'diet' | 'timeline'
  const [hasStarted, setHasStarted] = useState(false);

  // Auto-start recommendation synthesis upon opening tab if not yet generated
  useEffect(() => {
    if (user?.id && !hasStarted) {
      setHasStarted(true);
      if (historyData?.recommendations) {
        setRecs(historyData.recommendations);
      } else {
        runGeminiAnalysis();
      }
    }
  }, [user, historyData, hasStarted]);

  // Triggered when user clicks "Open AI Recommendations" button
  useEffect(() => {
    if (runTrigger && runTrigger > 0 && user?.id) {
      runGeminiAnalysis();
    }
  }, [runTrigger]);

  // Keep recs in sync if historyData updates externally
  useEffect(() => {
    if (historyData?.recommendations && !loading) {
      setRecs(historyData.recommendations);
    }
  }, [historyData?.recommendations]);

  const runGeminiAnalysis = async () => {
    if (!user?.id) return;
    setLoading(true);
    setError(null);
    try {
      const res = await triggerGeminiRecommendation(user.id);
      if (res && res.recommendations) {
        setRecs(res.recommendations);
        if (onRefreshHistory) {
          onRefreshHistory();
        }
      } else {
        setError('Failed to generate Gemini AI recommendations.');
      }
    } catch (err) {
      console.error('Failed to trigger Gemini API:', err);
      const msg = err.response?.data?.error || err.message || 'Gemini AI service error. Please verify backend connection.';
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const getUrgencyPill = (colorKey, label) => {
    let color = '#10b981';
    let bg = 'rgba(16, 185, 129, 0.12)';
    let border = 'rgba(16, 185, 129, 0.25)';

    if (colorKey === 'rose') {
      color = '#f43f5e';
      bg = 'rgba(244, 63, 94, 0.12)';
      border = 'rgba(244, 63, 94, 0.3)';
    } else if (colorKey === 'orange') {
      color = '#f97316';
      bg = 'rgba(249, 115, 22, 0.12)';
      border = 'rgba(249, 115, 22, 0.3)';
    } else if (colorKey === 'amber') {
      color = '#f59e0b';
      bg = 'rgba(245, 158, 11, 0.12)';
      border = 'rgba(245, 158, 11, 0.3)';
    }

    return (
      <span style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: '6px',
        padding: '5px 14px',
        borderRadius: '100px',
        fontSize: '12px',
        fontWeight: '700',
        color: color,
        background: bg,
        border: `1px solid ${border}`
      }}>
        <span style={{ width: '7px', height: '7px', borderRadius: '50%', background: color }} />
        {label}
      </span>
    );
  };

  if (!user) return null;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      
      {/* Top Hero Banner */}
      <div className="glass-card" style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '16px',
        padding: '24px 30px',
        background: 'linear-gradient(135deg, rgba(15, 23, 42, 0.7) 0%, rgba(6, 182, 212, 0.08) 100%)',
        border: '1px solid rgba(6, 182, 212, 0.25)',
        position: 'relative',
        overflow: 'hidden'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <div style={{
            width: '48px',
            height: '48px',
            borderRadius: '14px',
            background: 'linear-gradient(135deg, #06b6d4, #3b82f6)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 0 20px rgba(6, 182, 212, 0.4)'
          }}>
            <Sparkles size={24} color="#fff" />
          </div>
          <div>
            <h2 style={{ fontSize: '20px', fontWeight: '800', letterSpacing: '-0.5px', margin: 0 }}>
              AI Clinical Recommendation System
            </h2>
            <p style={{ fontSize: '13px', color: 'var(--text-secondary)', marginTop: '4px' }}>
              Personalized care intelligence powered strictly by Google Gemini AI, grounded in your CT scan history.
            </p>
          </div>
        </div>

        <button
          className="ai-analysis-btn"
          disabled={loading}
          onClick={runGeminiAnalysis}
          title="Run fresh clinical analysis with Google Gemini AI"
        >
          {loading ? (
            <>
              <div className="revolving-spinner-sm" />
              <span>Rendering Analysis...</span>
            </>
          ) : (
            <>
              <Sparkles size={16} />
              <span>{recs ? 'Re-run AI Analysis' : 'Run AI Analysis'}</span>
            </>
          )}
        </button>
      </div>

      {/* Rendering State: Revolving Circle UI */}
      {loading && (
        <RevolvingLoader
          badge="Google Gemini AI Pipeline"
          title="Synthesizing AI Clinical Intelligence..."
          subtitle="Analyzing verified sequential CT scan history, clinical risk trajectory, and formulating personalized patient guidance..."
          minHeight="360px"
        />
      )}

      {/* Error State */}
      {!loading && error && (
        <div className="glass-card" style={{
          padding: '36px 24px',
          textAlign: 'center',
          background: 'rgba(244, 63, 94, 0.05)',
          border: '1px solid rgba(244, 63, 94, 0.3)',
          borderRadius: '16px'
        }}>
          <AlertTriangle size={36} color="#f43f5e" style={{ margin: '0 auto 12px' }} />
          <h4 style={{ fontSize: '16px', fontWeight: '800', color: 'var(--text-primary)', marginBottom: '6px' }}>
            Gemini AI Analysis Failed
          </h4>
          <p style={{ fontSize: '13px', color: 'var(--text-secondary)', maxWidth: '440px', margin: '0 auto 20px' }}>
            {error}
          </p>
          <button
            className="predict-btn"
            onClick={runGeminiAnalysis}
            style={{ width: 'auto', padding: '10px 24px', fontSize: '13px', margin: '0 auto' }}
          >
            <RefreshCw size={15} />
            <span>Retry AI Analysis</span>
          </button>
        </div>
      )}

      {/* Main Pop-In Recommendation Dashboard */}
      {!loading && !error && recs && (
        <div className="pop-in-card" style={{
          background: 'rgba(15, 23, 42, 0.75)',
          backdropFilter: 'blur(20px)',
          border: '1px solid rgba(255, 255, 255, 0.08)',
          borderRadius: '16px',
          padding: '24px',
          boxShadow: '0 16px 48px rgba(0, 0, 0, 0.4)',
          position: 'relative',
          overflow: 'hidden'
        }}>
          {/* Subtle Accent Glow Bar */}
          <div style={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            height: '3px',
            background: recs.urgency_color === 'rose'
              ? 'linear-gradient(90deg, #f43f5e, #f97316)'
              : recs.urgency_color === 'orange'
              ? 'linear-gradient(90deg, #f97316, #f59e0b)'
              : recs.urgency_color === 'amber'
              ? 'linear-gradient(90deg, #f59e0b, #3b82f6)'
              : 'linear-gradient(90deg, #10b981, #06b6d4)'
          }} />

          {/* Top Status & Urgency Header */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '14px', marginBottom: '20px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <div style={{
                width: '40px',
                height: '40px',
                borderRadius: '10px',
                background: 'rgba(6, 182, 212, 0.12)',
                border: '1px solid rgba(6, 182, 212, 0.25)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}>
                <Sparkles size={20} color="var(--accent-cyan)" />
              </div>
              <div>
                <h3 style={{ fontSize: '17px', fontWeight: '800', color: 'var(--text-primary)', margin: 0 }}>
                  Personalized Care Plan
                </h3>
                <div style={{ fontSize: '12px', color: 'var(--accent-cyan)', fontWeight: '600', marginTop: '2px' }}>
                  {recs.powered_by || `Synthesized across ${recs.total_scans_analyzed} verified scan(s)`}
                </div>
              </div>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              {getUrgencyPill(recs.urgency_color, `Urgency: ${recs.urgency}`)}
            </div>
          </div>

          {/* Longitudinal Trajectory & Condition Pills */}
          <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap', marginBottom: '20px' }}>
            <span style={{
              background: 'rgba(255,255,255,0.04)',
              border: '1px solid rgba(255,255,255,0.08)',
              padding: '6px 14px',
              borderRadius: '10px',
              fontSize: '12.5px',
              color: 'var(--text-secondary)'
            }}>
              📈 <strong>Longitudinal Trajectory:</strong> {recs.risk_trend}
            </span>
            <span style={{
              background: 'rgba(255,255,255,0.04)',
              border: '1px solid rgba(255,255,255,0.08)',
              padding: '6px 14px',
              borderRadius: '10px',
              fontSize: '12.5px',
              color: recs.primary_condition.includes('Normal')
                ? 'var(--accent-emerald)'
                : (recs.primary_condition.includes('No Scan') || recs.total_scans_analyzed === 0)
                ? 'var(--accent-cyan)'
                : 'var(--accent-rose)',
              fontWeight: '600'
            }}>
              🏷️ <strong>Primary Condition:</strong> {recs.primary_condition}
            </span>
          </div>

          {/* Grounded Clinical Guidance Text */}
          <div style={{
            fontSize: '13.5px',
            color: 'var(--text-primary)',
            background: 'rgba(0, 0, 0, 0.35)',
            padding: '16px 18px',
            borderRadius: '12px',
            marginBottom: '20px',
            borderLeft: recs.total_scans_analyzed === 0 ? '4px solid var(--accent-amber)' : '4px solid var(--accent-cyan)',
            lineHeight: '1.6'
          }}>
            {recs.summary}
          </div>

          {/* Zero-Scan Callout Action Banner */}
          {recs.total_scans_analyzed === 0 && onNavigateToAnalysis && (
            <div style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              background: 'linear-gradient(135deg, rgba(6, 182, 212, 0.12) 0%, rgba(59, 130, 246, 0.1) 100%)',
              border: '1px solid rgba(6, 182, 212, 0.3)',
              borderRadius: '12px',
              padding: '14px 18px',
              marginBottom: '20px',
              gap: '12px',
              flexWrap: 'wrap'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <Info size={18} color="var(--accent-cyan)" />
                <span style={{ fontSize: '13px', color: 'var(--text-primary)' }}>
                  No medical scan records currently exist in your patient file to formulate a clinical diagnosis.
                </span>
              </div>
              <button
                onClick={onNavigateToAnalysis}
                className="ai-analysis-btn"
                style={{ padding: '7px 16px', fontSize: '12.5px' }}
              >
                Upload First CT Scan
              </button>
            </div>
          )}

          {/* Interactive Category Switcher Tabs */}
          <div style={{
            display: 'flex',
            gap: '8px',
            marginBottom: '20px',
            background: 'rgba(0,0,0,0.3)',
            padding: '4px',
            borderRadius: '12px',
            border: '1px solid rgba(255,255,255,0.05)'
          }}>
            <button
              onClick={() => setActiveCategory('clinical')}
              style={{
                flex: 1,
                padding: '10px 14px',
                borderRadius: '8px',
                border: 'none',
                background: activeCategory === 'clinical' ? 'rgba(59, 130, 246, 0.25)' : 'transparent',
                color: activeCategory === 'clinical' ? '#60a5fa' : 'var(--text-muted)',
                fontSize: '13px',
                fontWeight: '700',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '8px',
                transition: 'all 0.2s ease'
              }}
            >
              <Stethoscope size={16} />
              Clinical Steps
            </button>

            <button
              onClick={() => setActiveCategory('diet')}
              style={{
                flex: 1,
                padding: '10px 14px',
                borderRadius: '8px',
                border: 'none',
                background: activeCategory === 'diet' ? 'rgba(16, 185, 129, 0.25)' : 'transparent',
                color: activeCategory === 'diet' ? '#34d399' : 'var(--text-muted)',
                fontSize: '13px',
                fontWeight: '700',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '8px',
                transition: 'all 0.2s ease'
              }}
            >
              <Utensils size={16} />
              Diet & Hydration
            </button>

            <button
              onClick={() => setActiveCategory('timeline')}
              style={{
                flex: 1,
                padding: '10px 14px',
                borderRadius: '8px',
                border: 'none',
                background: activeCategory === 'timeline' ? 'rgba(245, 158, 11, 0.25)' : 'transparent',
                color: activeCategory === 'timeline' ? '#fbbf24' : 'var(--text-muted)',
                fontSize: '13px',
                fontWeight: '700',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '8px',
                transition: 'all 0.2s ease'
              }}
            >
              <Clock size={16} />
              Follow-Up & Notes
            </button>
          </div>

          {/* Active Tab Content Display */}
          <div style={{
            background: 'rgba(0, 0, 0, 0.25)',
            borderRadius: '12px',
            padding: '20px',
            border: '1px solid rgba(255, 255, 255, 0.05)'
          }}>
            {activeCategory === 'clinical' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                <h5 style={{ fontSize: '13px', fontWeight: '700', color: 'var(--text-secondary)', marginBottom: '4px' }}>
                  Recommended Action Items
                </h5>
                {recs.clinical_actions.map((act, idx) => (
                  <div key={idx} style={{ display: 'flex', alignItems: 'flex-start', gap: '12px', fontSize: '13.5px', color: 'var(--text-primary)' }}>
                    <div style={{ background: 'rgba(96, 165, 250, 0.15)', borderRadius: '6px', padding: '4px', marginTop: '2px', flexShrink: 0 }}>
                      <Check size={14} color="#60a5fa" />
                    </div>
                    <span style={{ lineHeight: '1.5' }}>{act}</span>
                  </div>
                ))}
              </div>
            )}

            {activeCategory === 'diet' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                <h5 style={{ fontSize: '13px', fontWeight: '700', color: 'var(--text-secondary)', marginBottom: '4px' }}>
                  Nutrition & Lifestyle Protocol
                </h5>
                {recs.dietary_lifestyle.map((diet, idx) => (
                  <div key={idx} style={{ display: 'flex', alignItems: 'flex-start', gap: '12px', fontSize: '13.5px', color: 'var(--text-primary)' }}>
                    <div style={{ background: 'rgba(52, 211, 153, 0.15)', borderRadius: '6px', padding: '4px', marginTop: '2px', flexShrink: 0 }}>
                      <Check size={14} color="#34d399" />
                    </div>
                    <span style={{ lineHeight: '1.5' }}>{diet}</span>
                  </div>
                ))}
              </div>
            )}

            {activeCategory === 'timeline' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                <div style={{
                  background: 'rgba(245, 158, 11, 0.1)',
                  border: '1px solid rgba(245, 158, 11, 0.25)',
                  borderRadius: '10px',
                  padding: '14px 16px',
                  fontSize: '13.5px',
                  color: '#fbbf24',
                  fontWeight: '600'
                }}>
                  <strong>Next Imaging Recommendation:</strong> {recs.follow_up_timeline}
                </div>
                <div style={{
                  fontSize: '12.5px',
                  color: 'var(--text-muted)',
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: '10px',
                  background: 'rgba(255, 255, 255, 0.02)',
                  padding: '12px',
                  borderRadius: '8px'
                }}>
                  <Info size={16} style={{ flexShrink: 0, marginTop: '2px', color: 'var(--accent-cyan)' }} />
                  <span>{recs.confidence_assessment}</span>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* If not started and no recs yet */}
      {!loading && !error && !recs && (
        <div className="glass-card" style={{
          textAlign: 'center',
          padding: '50px 20px',
          background: 'rgba(15, 23, 42, 0.5)'
        }}>
          <Sparkles size={36} color="var(--accent-cyan)" style={{ margin: '0 auto 14px', opacity: 0.8 }} />
          <h4 style={{ fontSize: '16px', fontWeight: '700', color: 'var(--text-primary)', marginBottom: '8px' }}>
            Awaiting AI Recommendation Analysis
          </h4>
          <p style={{ fontSize: '13px', color: 'var(--text-muted)', maxWidth: '360px', margin: '0 auto 20px' }}>
            Click <strong>Run AI Analysis</strong> to generate your tailored clinical care plan using Google Gemini.
          </p>
          <button
            className="ai-analysis-btn"
            onClick={runGeminiAnalysis}
          >
            <Sparkles size={16} />
            <span>Run AI Analysis</span>
          </button>
        </div>
      )}

    </div>
  );
};

export default RecommendationView;
