import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { FileText, Loader2 } from 'lucide-react';
import { PageHeader } from '../components/PageHeader';
import { useToast } from '../components/Toast';
import { useDocumentTitle } from '../hooks/useDocumentTitle';
import { appApi, type SessionSummary, type TranscriptLine } from '../services/app.api';
import './Sessions.css';

const clock = (seconds: number) => {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${String(s).padStart(2, '0')}`;
};

export function Sessions() {
  const { t } = useTranslation();
  useDocumentTitle(t('sessions.title'));
  const toast = useToast();

  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [selected, setSelected] = useState<number | null>(null);
  const [lines, setLines] = useState<TranscriptLine[]>([]);
  const [names, setNames] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(true);

  const fail = (err: unknown) => toast.error(err instanceof Error ? err.message : String(err));

  useEffect(() => {
    appApi
      .sessions()
      .then(list => {
        setSessions(list);
        if (list.length) setSelected(list[0].id);
      })
      .catch(fail)
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (selected === null) return;
    appApi
      .sessionLines(selected)
      .then(r => {
        setLines(r.lines);
        setNames(r.speakers);
      })
      .catch(fail);
  }, [selected]);

  // Speakers are identified by voice, not by name — the app never sees the participant list.
  // Naming them once here is what turns S1/S2 into a readable transcript.
  const saveName = async (code: string, name: string) => {
    if (selected === null) return;
    try {
      setNames(await appApi.setSpeakerNames(selected, { [code]: name }));
    } catch (err) {
      fail(err);
    }
  };

  const codes = [...new Set(lines.map(l => l.speaker))];
  const langs = [...new Set(lines.flatMap(l => Object.keys(l.translations)))];

  if (loading) {
    return (
      <div className="etable-page etable-loading">
        <Loader2 className="animate-spin" size={32} />
      </div>
    );
  }

  return (
    <div className="etable-page">
      <PageHeader title={t('sessions.title')} subtitle={t('sessions.subtitle')} />

      {sessions.length === 0 ? (
        <div className="sess-empty">
          <FileText size={32} strokeWidth={1} />
          <span>{t('sessions.empty')}</span>
        </div>
      ) : (
        <>
          <section className="etable-panel">
            <select className="sess-select" value={selected ?? ''} onChange={e => setSelected(Number(e.target.value))}>
              {sessions.map(s => (
                <option key={s.id} value={s.id}>
                  {s.started} — {t('sessions.lineCount', { count: s.lines })}
                </option>
              ))}
            </select>
          </section>

          {codes.length > 0 && (
            <section className="etable-panel">
              <h3 className="etable-panel-title">{t('sessions.speakers')}</h3>
              <p className="sess-hint">{t('sessions.speakersHint')}</p>
              <div className="sess-names">
                {codes.map(code => (
                  <label key={code} className="sess-name">
                    <span>{code}</span>
                    <input
                      defaultValue={names[code] ?? ''}
                      placeholder={t('sessions.namePlaceholder')}
                      onBlur={e => saveName(code, e.target.value)}
                    />
                  </label>
                ))}
              </div>
            </section>
          )}

          <section className="etable-panel">
            <h3 className="etable-panel-title">
              {t('sessions.transcript')}
              <span className="etable-count">{lines.length}</span>
            </h3>
            <div className="sess-lines">
              {lines.map(line => (
                <article key={line.id} className="sess-line">
                  <span className="sess-time">{clock(line.start)}</span>
                  <span className="sess-who">{names[line.speaker] || line.speaker}</span>
                  <div className="sess-body">
                    <p className="sess-source" lang={line.lang}>
                      {line.source}
                    </p>
                    {langs
                      .filter(l => line.translations[l])
                      .map(l => (
                        <p key={l} className="sess-translation" lang={l}>
                          {line.translations[l]}
                        </p>
                      ))}
                  </div>
                </article>
              ))}
            </div>
          </section>
        </>
      )}
    </div>
  );
}

export default Sessions;
