import React, { useState, useEffect } from 'react';
import { UploadCloud, Layers, LineChart } from 'lucide-react';
import Navbar from './components/Navbar';
import AuthPage from './components/AuthPage';
import ImageUploader from './components/ImageUploader';
import AnalysisResults from './components/AnalysisResults';
import GradcamVisualizer from './components/GradcamVisualizer';
import PatientDashboard from './components/PatientDashboard';
import { fetchUserHistory, predictImage } from './services/api';
import { logoutUser } from './services/api';

function App() {
  const [activeTab, setActiveTab] = useState('analysis');
  const [user, setUser] = useState(null);
  const [userHistory, setUserHistory] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [currentOriginalImage, setCurrentOriginalImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [toast, setToast] = useState(null);

  // Session is handled strictly without localStorage persistence per user request.

  const showToast = (message, type = 'success') => {
    setToast({ message, type });
    setTimeout(() => setToast(null), 4000);
  };

  const handleLoginSuccess = (userData) => {
    setUser(userData);
  };

  const handleLogout = async () => {
    try {
      await logoutUser();
    } catch (e) {
      console.error(e);
    }
    setUser(null);
    setUserHistory(null);
    setAnalysisResult(null);
    showToast('Signed out of Patient Portal.');
  };

  const loadHistory = async (userId) => {
    if (!userId) return;
    try {
      const history = await fetchUserHistory(userId);
      setUserHistory(history);
    } catch (err) {
      console.error('Failed to load user history:', err);
    }
  };

  useEffect(() => {
    if (user?.id) {
      loadHistory(user.id);
    }
  }, [user]);

  useEffect(() => {
    window.handleDeleteScan = async (scanId) => {
      if (!window.confirm("Are you sure you want to delete this scan record?")) return;
      try {
        const { deleteScan } = await import('./services/api');
        await deleteScan(scanId);
        showToast('Scan deleted successfully.');
        if (user?.id) loadHistory(user.id);
      } catch (err) {
        showToast('Failed to delete scan.', 'error');
      }
    };
    return () => { delete window.handleDeleteScan; };
  }, [user]);

  const handleAnalyze = async (base64Image) => {
    setLoading(true);
    setCurrentOriginalImage(`data:image/jpeg;base64,${base64Image}`);
    try {
      const data = await predictImage(base64Image, user ? user.id : null);
      if (data && data.length > 0) {
        const res = data[0];
        setAnalysisResult(res);
        showToast(`Diagnosed as: ${res.image} (${res.confidence.toFixed(1)}%)`);
        if (user?.id) {
          await loadHistory(user.id);
        }
      }
    } catch (err) {
      console.error(err);
      showToast('Scan analysis failed. Ensure backend API is running.', 'error');
    } finally {
      setLoading(false);
    }
  };

  if (!user) {
    return (
      <>
        <div className="bg-effects" />
        <div className="grid-overlay" />
        <AuthPage onLoginSuccess={handleLoginSuccess} showToast={showToast} />
        {toast && (
          <div className={`toast ${toast.type}`}>
            <span>{toast.message}</span>
          </div>
        )}
      </>
    );
  }

  return (
    <>
      <div className="bg-effects" />
      <div className="grid-overlay" />

      <div className="app-layout">
        {/* Sidebar */}
        <div className="sidebar">
          <div className="brand-icon" style={{ width: 40, height: 40, marginBottom: 40, fontSize: 18 }}>R</div>
          <div 
            className={`sidebar-icon ${activeTab === 'analysis' ? 'active' : ''}`} 
            onClick={() => setActiveTab('analysis')}
            title="Input & Analysis"
          >
            <UploadCloud size={24} />
          </div>
          <div 
            className={`sidebar-icon ${activeTab === 'gradcam' ? 'active' : ''}`} 
            onClick={() => setActiveTab('gradcam')}
            title="Grad-CAM"
          >
            <Layers size={24} />
          </div>
          <div 
            className={`sidebar-icon ${activeTab === 'history' ? 'active' : ''}`} 
            onClick={() => setActiveTab('history')}
            title="History Chart"
          >
            <LineChart size={24} />
          </div>
        </div>

        {/* Main Content Area */}
        <div className="main-content">
          <div className="content-container">
            <Navbar user={user} onLogout={handleLogout} />

            <div style={{ display: activeTab === 'analysis' ? 'block' : 'none' }}>
              {/* Hero */}
              <header className="hero">
                <h1>Welcome, <span>{user.name}</span></h1>
                <p>
                  Upload your abdominal CT scans to get deep learning disease classification,
                  Grad-CAM explainability heatmaps, and track your health recovery trend.
                </p>
                <div className="class-chips">
                  <span className="chip normal">Normal</span>
                  <span className="chip cyst">Cyst</span>
                  <span className="chip stone">Stone</span>
                  <span className="chip tumor">Tumor</span>
                </div>
              </header>

              {/* Scanner Grid */}
              <div className="main-grid">
                <div>
                  <ImageUploader onAnalyze={handleAnalyze} loading={loading} />
                </div>
                <div>
                  <AnalysisResults result={analysisResult} />
                </div>
              </div>
            </div>

            <div style={{ display: activeTab === 'gradcam' ? 'block' : 'none', marginBottom: '36px' }}>
              {analysisResult ? (
                <GradcamVisualizer result={analysisResult} originalImage={currentOriginalImage} />
              ) : (
                <div className="glass-card" style={{ textAlign: 'center', padding: '60px' }}>
                  <p style={{ color: 'var(--text-muted)' }}>No analysis result available yet. Please upload and analyze a scan first.</p>
                </div>
              )}
            </div>

            <div style={{ display: activeTab === 'history' ? 'block' : 'none' }}>
              <PatientDashboard user={user} historyData={userHistory} />
            </div>
          </div>
        </div>
      </div>

      {toast && (
        <div className={`toast ${toast.type}`}>
          <span>{toast.message}</span>
        </div>
      )}
    </>
  );
}

export default App;
