import type { ReactNode } from 'react';
import './PageHeader.css';

interface PageHeaderProps {
  title: string;
  subtitle?: string;
  badge?: ReactNode;
  actions?: ReactNode;
}

/**
 * Shared page header component for consistent styling across all pages.
 *
 * @example
 * // Simple usage
 * <PageHeader title="Subtitles" subtitle="How subtitles are laid out on the TV." />
 *
 * @example
 * // With badge and actions
 * <PageHeader
 *   title="Sessions"
 *   badge={<StatusBadge status="connected" />}
 *   subtitle="Transcripts and translations from past meetings."
 *   actions={<button>Import a recording</button>}
 * />
 */
export function PageHeader({ title, subtitle, badge, actions }: PageHeaderProps) {
  return (
    <header className="page-header">
      <div className="page-header__title-group">
        <h1>{title}</h1>
        {badge && <span className="page-header__badge">{badge}</span>}
      </div>
      {actions && <div className="page-header__actions">{actions}</div>}
      {subtitle && <p className="page-header__subtitle">{subtitle}</p>}
    </header>
  );
}
