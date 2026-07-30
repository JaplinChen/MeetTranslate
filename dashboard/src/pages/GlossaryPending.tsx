import { useTranslation } from 'react-i18next';
import { Check, Trash2 } from 'lucide-react';
import type { PendingGlossaryTerm } from '../services/api';

interface Props {
  pending: PendingGlossaryTerm[];
  canWrite: boolean;
  busy: boolean;
  onApprove: (id: number) => void;
  onReject: (id: number) => void;
}

export function GlossaryPending({ pending, canWrite, busy, onApprove, onReject }: Props) {
  const { t } = useTranslation();
  if (pending.length === 0) return null;
  return (
    <section className="etable-panel etable-panel--pending">
      <h3 className="etable-panel-title">
        {t('glossary.pendingTitle')}
        <span className="etable-count">{pending.length}</span>
      </h3>
      <div className="etable-list">
        {pending.map(p => (
          <div key={p.id} className="etable-item etable-item--pending">
            <span className="etable-src">{p.zh}</span>
            <span className="etable-arrow">→</span>
            <span className="etable-tgt">{p.vi}</span>
            <span className="etable-pending-meta">
              {t('glossary.suggestedBy', { name: p.suggestedBy })}
              {' · '}
              {new Date(p.at).toLocaleString()}
            </span>
            {canWrite && (
              <div className="etable-row-actions">
                <button
                  className="etable-del"
                  onClick={() => onApprove(p.id)}
                  disabled={busy}
                  title={t('glossary.approve')}
                >
                  <Check size={16} />
                </button>
                <button
                  className="etable-del"
                  onClick={() => onReject(p.id)}
                  disabled={busy}
                  title={t('glossary.reject')}
                >
                  <Trash2 size={16} />
                </button>
              </div>
            )}
          </div>
        ))}
      </div>
    </section>
  );
}
