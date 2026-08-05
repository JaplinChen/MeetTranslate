import { useCallback, useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { FileText, Upload } from 'lucide-react';
import { PageHeader } from '../components/PageHeader';
import { PageSkeleton } from '../components/PageSkeleton';
import { TranscriptRow } from '../components/sessions/TranscriptRow';
import { useToast } from '../components/Toast';
import { useDocumentTitle } from '../hooks/useDocumentTitle';
import { appApi, type RefineState, type SessionSummary, type TranscriptLine } from '../services/app.api';
import { API_BASE_URL } from '../services/http';
import './Sessions.css';
import './Sessions.refine.css';

// How often to re-check a session that is still being refined. The pass takes minutes, so this is
// about noticing it finished rather than tracking progress, and it stops the moment it has.
const REFINE_POLL_MS = 5000;

// One meeting is an import control, up to 35 speaker fields and ~950 transcript rows. Stacked they
// are one column metres long, where naming a speaker means scrolling past the transcript to find
// the field and scrolling back to see whether it took. Each is its own view; the session picker
// stays outside them because both of the other two are about whichever session it points at.
const TABS = ['import', 'speakers', 'transcript'] as const;
type Tab = (typeof TABS)[number];

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
  const [tab, setTab] = useState<Tab>('transcript');
  const [playing, setPlaying] = useState<number | null>(null);
  const tablistRef = useRef<HTMLDivElement>(null);
  const player = useRef<HTMLAudioElement | null>(null);

  const fail = (err: unknown) => toast.error(err instanceof Error ? err.message : String(err));

  const current = sessions.find(s => s.id === selected);
  const refine: RefineState = current?.refine.state ?? 'idle';
  // Playing a line, hearing a speaker and re-deriving the transcript all read the recording. When
  // it is gone they all fail the same way, so the page says so once instead of per click.
  const hasRecording = current?.hasRecording ?? true;
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

  // Correcting a line is a judgement about whether the text matches what was said, so the audio has
  // to be reachable from the line. One player for the whole transcript rather than one per row:
  // clicking a second line replaces what is playing, which is also the behaviour you want.
  const playLine = (lineId: number) => {
    if (selected === null || !hasRecording) return;
    const audio = (player.current ??= new Audio());
    if (playing === lineId) {
      audio.pause();
      setPlaying(null);
      return;
    }
    audio.src = `${API_BASE_URL}/sessions/${selected}/lines/${lineId}/clip`;
    audio.onended = () => setPlaying(null);
    audio.onerror = () => {
      setPlaying(null);
      toast.error(t('sessions.playFailed'));
    };
    void audio.play().catch(() => {});
    setPlaying(lineId);
  };

  // Switching session or tab leaves a clip playing over a transcript that is no longer on screen.
  useEffect(() => {
    player.current?.pause();
    setPlaying(null);
  }, [selected, tab]);

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

  // With nothing imported there is no session for the other two to be about, so the choice is not
  // offered rather than offered and empty.
  const hasSessions = sessions.length > 0;
  const active: Tab = hasSessions ? tab : 'import';

  // Left/Right move between tabs, which is what a tablist is expected to do once its buttons claim
  // role="tab" — without it the role announces an interaction the keyboard cannot perform.
  const onTabKeys = (event: React.KeyboardEvent) => {
    const step = event.key === 'ArrowRight' ? 1 : event.key === 'ArrowLeft' ? -1 : 0;
    if (!step) return;
    event.preventDefault();
    const enabled = TABS.filter(id => id === 'import' || hasSessions);
    const next = enabled[(enabled.indexOf(active) + step + enabled.length) % enabled.length];
    setTab(next);
    tablistRef.current?.querySelector<HTMLButtonElement>(`#sess-tab-${next}`)?.focus();
  };

  return (
    <div className="etable-page sess-page">
      <PageHeader title={t('sessions.title')} subtitle={t('sessions.subtitle')} />

      {hasSessions && (
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
      )}

      <div className="sess-tabs" role="tablist" aria-label={t('sessions.title')} ref={tablistRef} onKeyDown={onTabKeys}>
        {TABS.map(id => (
          <button
            key={id}
            id={`sess-tab-${id}`}
            role="tab"
            type="button"
            aria-selected={active === id}
            aria-controls={`sess-panel-${id}`}
            // Only the active tab is in the tab order; arrows move within the list. Five stops for
            // three tabs is what makes a tablist tedious to tab past.
            tabIndex={active === id ? 0 : -1}
            disabled={id !== 'import' && !hasSessions}
            className={`sess-tab ${active === id ? 'active' : ''}`}
            onClick={() => setTab(id)}
          >
            {t(`sessions.${id}`)}
            {id === 'transcript' && hasSessions && <span className="etable-count">{lines.length}</span>}
          </button>
        ))}
      </div>

      {active === 'import' && (
        <section className="etable-panel" role="tabpanel" id="sess-panel-import" aria-labelledby="sess-tab-import">
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
          {!hasSessions && (
            <div className="sess-empty">
              <FileText size={32} strokeWidth={1} />
              <span>{t('sessions.empty')}</span>
            </div>
          )}
        </section>
      )}

      {active === 'speakers' && (
        <section className="etable-panel" role="tabpanel" id="sess-panel-speakers" aria-labelledby="sess-tab-speakers">
          <p className="sess-hint">{t('sessions.speakersHint')}</p>
          {!hasRecording && <p className="sess-no-audio">{t('sessions.noRecording')}</p>}
          <div className="sess-names">
            {codes.map(code => (
              <div key={code} className="sess-name">
                <label className="sess-name-field">
                  <span>{code}</span>
                  <input
                    defaultValue={names[code] ?? ''}
                    placeholder={t('sessions.namePlaceholder')}
                    onBlur={e => saveName(code, e.target.value)}
                  />
                </label>
                {/* preload="none" because a meeting can have 35 of these and none of them is
                    wanted until someone clicks. Omitted entirely when the recording is gone —
                    35 players that can only fail are worse than none. */}
                {hasRecording && (
                  <audio
                    className="sess-clip"
                    controls
                    preload="none"
                    aria-label={t('sessions.clipLabel', { code })}
                    src={`${API_BASE_URL}/sessions/${selected}/speakers/${encodeURIComponent(code)}/clip`}
                  />
                )}
              </div>
            ))}
          </div>
        </section>
      )}

      {active === 'transcript' && (
        <section className="etable-panel" role="tabpanel" id="sess-panel-transcript" aria-labelledby="sess-tab-transcript">
          <h3 className="etable-panel-title">
            {t('sessions.transcript')}
            <span className="etable-count">{lines.length}</span>
            {refine !== 'idle' && (
              <span className={`sess-refine sess-refine-${refine}`}>{refineLabel[refine]}</span>
            )}
          </h3>
          {locked && <p className="sess-hint">{t('sessions.refiningHint')}</p>}
          {!hasRecording && <p className="sess-no-audio">{t('sessions.noRecording')}</p>}
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
                playing={playing}
                playable={hasRecording}
                onDraft={setEditing}
                onSave={saveLine}
                onRerun={rerunLine}
                onPlay={playLine}
              />
            ))}
          </div>
        </section>
      )}
    </div>
  );
}

export default Sessions;
