import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Brain, Trash2 } from 'lucide-react';
import { PageHeader } from '../components/PageHeader';
import { PageSkeleton } from '../components/PageSkeleton';
import { useToast } from '../components/Toast';
import { useDocumentTitle } from '../hooks/useDocumentTitle';
import { appApi, type KnownSpeaker, type LearnedCorrection } from '../services/app.api';
import './Learned.css';

/**
 * What the room has picked up on its own: voices it can now name, and mistakes it will not repeat.
 *
 * Both are learned from ordinary use rather than configured — naming a speaker, correcting a line.
 * The reason this page exists is that learning silently is only acceptable if it can be inspected
 * and undone: a voiceprint attached to the wrong person, or a correction learned from a typo,
 * would otherwise keep applying with nowhere to see it.
 */
export function Learned() {
  const { t } = useTranslation();
  useDocumentTitle(t('learned.title'));
  const toast = useToast();

  const [speakers, setSpeakers] = useState<KnownSpeaker[]>([]);
  const [corrections, setCorrections] = useState<LearnedCorrection[]>([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);

  const fail = (err: unknown) => toast.error(err instanceof Error ? err.message : String(err));

  useEffect(() => {
    let alive = true;
    Promise.all([appApi.knownSpeakers(), appApi.corrections()])
      .then(([voices, fixes]) => {
        if (!alive) return;
        setSpeakers(voices);
        setCorrections(fixes);
      })
      .catch(err => alive && fail(err))
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
  }, []);

  const forgetSpeaker = async (name: string) => {
    setBusy(true);
    try {
      setSpeakers(await appApi.forgetSpeaker(name));
    } catch (err) {
      fail(err);
    } finally {
      setBusy(false);
    }
  };

  const forgetCorrection = async (wrong: string) => {
    setBusy(true);
    try {
      setCorrections(await appApi.forgetCorrection(wrong));
    } catch (err) {
      fail(err);
    } finally {
      setBusy(false);
    }
  };

  if (loading) {
    return <PageSkeleton />;
  }

  return (
    <div className="etable-page">
      <PageHeader title={t('learned.title')} subtitle={t('learned.subtitle')} />

      <section className="etable-panel">
        <h3 className="etable-panel-title">
          {t('learned.voices')}
          <span className="etable-count">{speakers.length}</span>
        </h3>
        <p className="learned-note">{t('learned.voicesNote')}</p>
        {speakers.length === 0 ? (
          <p className="learned-empty">{t('learned.noVoices')}</p>
        ) : (
          <ul className="learned-list">
            {speakers.map(s => (
              <li key={s.name} className="learned-row">
                <span className="learned-name">{s.name}</span>
                <button
                  className="learned-forget"
                  disabled={busy}
                  title={t('learned.forget')}
                  onClick={() => forgetSpeaker(s.name)}
                >
                  <Trash2 size={16} />
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="etable-panel">
        <h3 className="etable-panel-title">
          {t('learned.corrections')}
          <span className="etable-count">{corrections.length}</span>
        </h3>
        <p className="learned-note">{t('learned.correctionsNote')}</p>
        {corrections.length === 0 ? (
          <p className="learned-empty">{t('learned.noCorrections')}</p>
        ) : (
          <ul className="learned-list">
            {corrections.map(c => (
              <li key={c.wrong} className="learned-row">
                <span className="learned-wrong">{c.wrong}</span>
                <Brain className="learned-arrow" size={14} />
                <span className="learned-right">{c.right}</span>
                <button
                  className="learned-forget"
                  disabled={busy}
                  title={t('learned.forget')}
                  onClick={() => forgetCorrection(c.wrong)}
                >
                  <Trash2 size={16} />
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
