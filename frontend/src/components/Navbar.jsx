import React from 'react';
import { Activity, LogOut, User } from 'lucide-react';

const Navbar = ({ user, onLogout }) => {
  return (
    <nav className="navbar">
      <div className="navbar-brand">
        <div className="brand-icon">R</div>
        <div>
          <div className="brand-name">Renalvision</div>
          <div className="brand-tag">Deep Learning Diagnostics</div>
        </div>
      </div>

      <div className="nav-right">
        {user && (
          <>
            <div className="nav-user">
              <User size={14} className="text-blue-400" />
              <span>{user.name || user.username}</span>
            </div>
            <button onClick={onLogout} className="logout-btn" title="Sign out of portal">
              <LogOut size={14} style={{ display: 'inline', marginRight: '6px' }} />
              Logout
            </button>
          </>
        )}
      </div>
    </nav>
  );
};

export default Navbar;
