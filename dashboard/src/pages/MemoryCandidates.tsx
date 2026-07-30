import { useTranslation } from 'react-i18next';
import { Check, Trash2 } from 'lucide-react';
import type { TranslationCandidate } from '../services/api';

interface Props {
  candidates: TranslationCandidate[];
  canWrite: boolean;
  busy: boolean;
  onApprove: (id: number) => void;
  onDismiss: (id: number) => void;
}

export function MemoryCandidates({ candidates, canWrite, busy, onApprove, onDismiss }: Props) {
  const { t } = useTranslation();
  if (candidates.length === 0) return null;
  return (
    <section className="etable-panel etable-panel--pending">
      {/* No panel title: the glossary tab bar already shows this count. */}
      <div className="etable-list">
        {candidates.map(c => (
          <div key={c.id} className="etable-item etable-item--pending">
            <span className="etable-src">{c.source}</span>
            <span className="etable-arrow">→</span>
            <span className="etable-tgt">{c.translated}</span>
            <span className="etable-pending-meta">{t('glossary.seenCount', { count: c.count })}</span>
            {canWrite && (
              <div className="etable-row-actions">
                <button
                  className="etable-del"
                  onClick={() => onApprove(c.id)}
                  disabled={busy}
                  title={t('glossary.approve')}
                >
                  <Check size={16} />
                </button>
                <button
                  className="etable-del"
                  onClick={() => onDismiss(c.id)}
                  disabled={busy}
                  title={t('glossary.dismiss')}
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
