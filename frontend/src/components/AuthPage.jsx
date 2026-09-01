import React, { useState } from 'react';
import { User, Key, ShieldCheck, UserPlus } from 'lucide-react';
import { loginUser, registerUser } from '../services/api';

const AuthPage = ({ onLoginSuccess, showToast }) => {
  const [isRegistering, setIsRegistering] = useState(false);
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!username.trim() || !password.trim()) {
      showToast('Please enter your username and password.', 'error');
      return;
    }

    setLoading(true);
    try {
      if (isRegistering) {
        if (!name.trim() || !email.trim()) {
          showToast('Please enter your full name and email.', 'error');
          setLoading(false);
          return;
        }
        const data = await registerUser(username, email, password, name);
        showToast(`Account created successfully! Welcome, ${data.user.name}.`);
        onLoginSuccess(data.user);
      } else {
        const data = await loginUser(username, password);
        showToast(`Welcome back, ${data.user.name}!`);
        onLoginSuccess(data.user);
      }
    } catch (err) {
      console.error(err);
      const errMsg = err.response?.data?.error || 'Authentication failed. Please check credentials.';
      showToast(errMsg, 'error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-container">
      <div className="auth-card">
        <div className="auth-header">
          <div className="auth-icon">🔬</div>
          <h2 className="auth-title">Patient Portal</h2>
          <p className="auth-subtitle">
            {isRegistering ? 'Create your personal account to track CT scan history' : 'Sign in to access your personal CT scan diagnostics'}
          </p>
        </div>

        {/* Auth Toggle Tabs */}
        <div style={{ display: 'flex', background: 'rgba(0,0,0,0.3)', padding: '4px', borderRadius: '10px', marginBottom: '24px' }}>
          <button
            type="button"
            style={{
              flex: 1,
              padding: '8px',
              borderRadius: '8px',
              border: 'none',
              fontSize: '13px',
              fontWeight: '700',
              cursor: 'pointer',
              background: !isRegistering ? 'var(--accent-blue)' : 'transparent',
              color: !isRegistering ? 'white' : 'var(--text-secondary)'
            }}
            onClick={() => setIsRegistering(false)}
          >
            Sign In
          </button>
          <button
            type="button"
            style={{
              flex: 1,
              padding: '8px',
              borderRadius: '8px',
              border: 'none',
              fontSize: '13px',
              fontWeight: '700',
              cursor: 'pointer',
              background: isRegistering ? 'var(--accent-blue)' : 'transparent',
              color: isRegistering ? 'white' : 'var(--text-secondary)'
            }}
            onClick={() => setIsRegistering(true)}
          >
            Create Account
          </button>
        </div>

        <form onSubmit={handleSubmit}>
          {isRegistering && (
            <>
              <div className="form-group">
                <label style={{ fontSize: '12px', fontWeight: '600', color: 'var(--text-secondary)' }}>
                  Full Name
                </label>
                <input
                  type="text"
                  className="form-input"
                  placeholder="e.g. Alex Johnson"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  required
                />
              </div>
              <div className="form-group">
                <label style={{ fontSize: '12px', fontWeight: '600', color: 'var(--text-secondary)' }}>
                  Email Address
                </label>
                <input
                  type="email"
                  className="form-input"
                  placeholder="e.g. alex@example.com"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                />
              </div>
            </>
          )}

          <div className="form-group">
            <label style={{ fontSize: '12px', fontWeight: '600', color: 'var(--text-secondary)' }}>
              Username
            </label>
            <div style={{ position: 'relative' }}>
              <input
                type="text"
                className="form-input"
                style={{ width: '100%', paddingLeft: '36px' }}
                placeholder="Enter username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                required
              />
              <User size={16} style={{ position: 'absolute', left: '12px', top: '12px', color: 'var(--text-muted)' }} />
            </div>
          </div>

          <div className="form-group">
            <label style={{ fontSize: '12px', fontWeight: '600', color: 'var(--text-secondary)' }}>
              Password
            </label>
            <div style={{ position: 'relative' }}>
              <input
                type="password"
                className="form-input"
                style={{ width: '100%', paddingLeft: '36px' }}
                placeholder="Enter password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
              <Key size={16} style={{ position: 'absolute', left: '12px', top: '12px', color: 'var(--text-muted)' }} />
            </div>
          </div>

          <button
            type="submit"
            className="predict-btn"
            disabled={loading}
            style={{ marginTop: '24px' }}
          >
            {loading ? (
              <span>Processing...</span>
            ) : isRegistering ? (
              <>
                <UserPlus size={18} />
                <span>Register & Sign In</span>
              </>
            ) : (
              <>
                <ShieldCheck size={18} />
                <span>Sign In to Portal</span>
              </>
            )}
          </button>
        </form>

        <div style={{ marginTop: '20px', textAlign: 'center', fontSize: '12px', color: 'var(--text-muted)' }}>
          🔒 Private & Confidential Personal Health Records
        </div>
      </div>
    </div>
  );
};

export default AuthPage;
