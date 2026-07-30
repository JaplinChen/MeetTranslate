import { NavLink, Outlet } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { KeyRound, Bot } from 'lucide-react';
import './Settings.css';

const settingsNavItems = [
  { to: 'llm', icon: Bot, key: 'llm' as const },
  { to: 'keyproxy', icon: KeyRound, key: 'keyproxy' as const },
];

export function Settings() {
  const { t } = useTranslation();

  return (
    <div className="settings-layout">
      <nav className="settings-nav">
        <span className="settings-nav-title">{t('nav.settings')}</span>
        {settingsNavItems.map(({ to, icon: Icon, key }) => (
          <NavLink key={to} to={to} className={({ isActive }) => `settings-nav-item ${isActive ? 'active' : ''}`}>
            <Icon size={18} />
            <span>{t(`nav.${key}`)}</span>
          </NavLink>
        ))}
      </nav>
      <div className="settings-content">
        <Outlet />
      </div>
    </div>
  );
}

export default Settings;
