import React, { useState } from 'react';
import { loginAdmin } from '../../services/api';
import { KeyRound } from 'lucide-react';

interface AdminLoginBarProps {
  onShowToast: (msg: string, type?: 'error' | 'success') => void;
}

export const AdminLoginBar: React.FC<AdminLoginBarProps> = ({ onShowToast }) => {
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);

  const handleLogin = async () => {
    if (!password.trim() || loading) return;
    setLoading(true);
    try {
      const data = await loginAdmin(password.trim());
      localStorage.setItem('qnu_admin_token', data.token);
      setPassword('');
      onShowToast('Đăng nhập quản trị thành công!', 'success');
      window.open('/admin', '_blank');
    } catch (err: any) {
      onShowToast(err.message || 'Sai mật khẩu!', 'error');
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleLogin();
    }
  };

  return (
    <div className="sidebar-sticky">
      <div className="admin-section">
        <div className="admin-login-row">
          <input
            type="password"
            className="admin-input"
            placeholder="Mật khẩu Admin..."
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={loading}
          />
          <button
            type="button"
            className="admin-btn primary"
            onClick={handleLogin}
            disabled={loading || !password.trim()}
            style={{ display: 'inline-flex', alignItems: 'center', gap: '4px' }}
          >
            <KeyRound size={13} />
            <span>{loading ? '...' : 'Admin'}</span>
          </button>
        </div>
      </div>
    </div>
  );
};
