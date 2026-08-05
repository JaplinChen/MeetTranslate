import { useCallback, useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { FileText, Upload } from 'lucide-react';
import { PageHeader } from '../components/PageHeader';
import { PageSkeleton } from '../components/PageSkeleton';
import { TranscriptRow } from '../components/sessions/TranscriptRow';
import { useToast } from '../components/Toast';
import { useDocumentTitle } from '../hooks/useDocumentTitle';
import { appApi, type RefineState, type SessionSummary, type TranscriptLine } from '../services/app.api';
import './Sessions.css';
import './Sessions.refine.css';

// How often to re-check a session that is still being refined. The pass takes minutes, so this is
// about noticing it finished rather than tracking progress, and it stops the moment it has.
const REFINE_POLL_MS = 5000;

export function Sessions() {
  const { t } = useTranslation();
  useDocumentTitle(t('sessions.title'));
  const toast = useToast();

  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [selected, setSelected] = useState<number | null>(null);
  const [lines, setLines] = useState<TranscriptLine[]>([]);
  const [names, setNames] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(true);
  const [editing, setEditing] = useState<{ id: number; text: string } | null>(null);
  const [importing, setImporting] = useState(false);
  const [rerunning, setRerunning] = useState<number | null>(null);

  const fail = (err: unknown) => toast.error(err instanceof Error ? err.message : String(err));

  const refine: RefineState = sessions.find(s => s.id === selected)?.refine.state ?? 'idle';
  // The pass calls replace_lines, which drops every line and writes new ones with new ids. An edit
  // saved during that window is silently discarded while the screen shows it saved, so editing is
  // closed rather than left to look like it worked.
  const locked = refine === 'refining';

  const loadLines = useCallback((id: number) => {
    appApi
      .sessionLines(id)
      .then(r => {
        setLines(r.lines);
        setNames(r.speakers);
      })
      .catch(fail);
  }, []);

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
    if (selected !== null) loadLines(selected);
  }, [selected, loadLines]);

  // Poll only while something is actually being refined, and stop as soon as it is not. Without
  // this the chip would say "refining" until someone reloaded the page by hand.
  const wasRefining = useRef(false);
  useEffect(() => {
    if (refine !== 'refining') {
      if (wasRefining.current && selected !== null) {
        wasRefining.current = false;
        toast.success(t('sessions.refineDone'));
        loadLines(selected);
      }
      return;
    }
    wasRefining.current = true;
    const timer = window.setInterval(() => {
      appApi.sessions().then(setSessions).catch(() => {});
    }, REFINE_POLL_MS);
    return () => window.clearInterval(timer);
  }, [refine, selected, loadLines, toast, t]);

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

  // Correcting a line is the only ground truth the system gets: someone who was in the room
  // saying what was actually said. The backend learns the pair and applies it from then on.
  //
  // A textarea rather than contentEditable: this transcript is mostly Chinese, and an IME
  // composing inside a contentEditable fires input and blur events mid-character.
  const saveLine = async (lineId: number, source: string, previous: string) => {
    setEditing(null);
    if (selected === null || source.trim() === previous || !source.trim()) return;
    try {
      const r = await appApi.setLineSource(selected, lineId, source.trim());
      setLines(r.lines);
    } catch (err) {
      fail(err);
    }
  };

  // A recording made elsewhere teaches the same things a live capture does, once it is a session:
  // names attach to voices, corrections attach to lines.
  const importRecording = async (file: File) => {
    setImporting(true);
    try {
      const added = await appApi.importRecording(file);
      setSessions(await appApi.sessions());
      setSelected(added.id);
    } catch (err) {
      fail(err);
    } finally {
      setImporting(false);
    }
  };

  // Re-running is per line rather than per transcript: a failure is usually one utterance the
  // decoder gave up on, and re-running the whole meeting to recover it is not a proportionate ask.
  const rerunLine = async (lineId: number) => {
    if (selected === null) return;
    setRerunning(lineId);
    try {
      const r = await appApi.rerunLine(selected, lineId);
      setLines(r.lines);
      setNames(r.speakers);
      if (r.status !== 'ok') toast.error(t(`sessions.${r.status === 'asr_failed' ? 'lineFailedAsr' : 'lineFailedTranslate'}`));
    } catch (err) {
      fail(err);
    } finally {
      setRerunning(null);
    }
  };

  const codes = [...new Set(lines.map(l => l.speaker))];
  const langs = [...new Set(lines.flatMap(l => Object.keys(l.translations)))];
  const failed = lines.filter(l => l.status !== 'ok');
  const refineLabel: Partial<Record<RefineState, string>> = {
    refining: t('sessions.refining'),
    refined: t('sessions.refined'),
    failed: t('sessions.refineFailed'),
    cancelled: t('sessions.refineCancelled'),
  };

  if (loading) {
    return <PageSkeleton rows={4} />;
  }

  return (
    <div className="etable-page sess-page">
      <PageHeader title={t('sessions.title')} subtitle={t('sessions.subtitle')} />

      <section className="etable-panel">
        <h3 className="etable-panel-title">{t('sessions.import')}</h3>
        <p className="sess-hint">{t('sessions.importHint')}</p>
        <label className="sess-import">
          <Upload size={16} />
          <span>{importing ? t('sessions.importing') : t('sessions.importPick')}</span>
          <input
            type="file"
            accept="video/*,audio/*"
            disabled={importing}
            onChange={e => {
              const file = e.target.files?.[0];
              e.target.value = '';
              if (file) importRecording(file);
            }}
          />
        </label>
      </section>

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
                  {s.refine.state === 'refining' ? ` · ${t('sessions.refining')}` : ''}
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
              {refine !== 'idle' && (
                <span className={`sess-refine sess-refine-${refine}`}>{refineLabel[refine]}</span>
              )}
            </h3>
            {locked && <p className="sess-hint">{t('sessions.refiningHint')}</p>}
            {failed.length > 0 && (
              // Aggregated as well as marked inline: a two-hour meeting failing 5% is forty-odd
              // marks scattered through the transcript, and nobody finds those by scrolling.
              <p className="sess-failed-summary">{t('sessions.failedCount', { count: failed.length })}</p>
            )}
            <div className="sess-lines">
              {lines.map(line => (
                <TranscriptRow
                  key={line.id}
                  line={line}
                  speaker={names[line.speaker] || line.speaker}
                  langs={langs}
                  locked={locked}
                  draft={editing}
                  rerunning={rerunning}
                  onDraft={setEditing}
                  onSave={saveLine}
                  onRerun={rerunLine}
                />
              ))}
            </div>
          </section>
        </>
      )}
    </div>
  );
}

export default Sessions;
