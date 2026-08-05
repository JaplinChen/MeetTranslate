import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { Play, RotateCw, Square } from 'lucide-react';
import type { TranscriptLine } from '../../services/app.api';

const clock = (seconds: number) => {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${String(s).padStart(2, '0')}`;
};

interface Props {
  line: TranscriptLine;
  speaker: string; // resolved display name, or the S1/S2 code when nobody has named the voice
  langs: string[]; // every target language in this transcript, so rows render them in one order
  locked: boolean; // a refine pass is running; see the comment on `draftText`
  // Per row, not per transcript: a row that is handed "which line is playing" re-renders whenever
  // any other line starts playing, and there are 943 of them.
  draftText: string | null; // this row's edit in progress, or null when it is not being edited
  isRerunning: boolean;
  rerunBlocked: boolean; // some other line is re-running, so this button waits its turn
  isPlaying: boolean;
  playable: boolean; // false once the session's recording is gone from disk
  onDraft: (draft: { id: number; text: string } | null) => void;
  onSave: (lineId: number, source: string, previous: string) => void;
  onRerun: (lineId: number) => void;
  onPlay: (lineId: number) => void;
}

function Row({ line, speaker, langs, locked, draftText, isRerunning, rerunBlocked, isPlaying, playable, onDraft, onSave, onRerun, onPlay }: Props) {
  const { t } = useTranslation();
  const editing = draftText === null ? null : { id: line.id, text: draftText };

  return (
    <article className={`sess-line${line.status === 'ok' ? '' : ' sess-line-failed'}`}>
      <div className="sess-time">
        {/* One button per line, but one <audio> for the whole transcript — 943 media elements is
            not a price worth paying for a control that plays one thing at a time. */}
        <button
          type="button"
          className="sess-play"
          disabled={!playable}
          title={!playable ? t('sessions.noRecording') : isPlaying ? t('sessions.stopLine') : t('sessions.playLine')}
          aria-label={!playable ? t('sessions.noRecording') : isPlaying ? t('sessions.stopLine') : t('sessions.playLine')}
          onClick={() => onPlay(line.id)}
        >
          {isPlaying ? <Square size={11} /> : <Play size={11} />}
        </button>
        <span>{clock(line.start)}</span>
      </div>
      <span className="sess-who">{speaker}</span>
      <div className="sess-body">
        {line.status !== 'ok' && (
          <div className="sess-status">
            <span className="sess-badge">
              {t(line.status === 'asr_failed' ? 'sessions.lineFailedAsr' : 'sessions.lineFailedTranslate')}
            </span>
            <button
              type="button"
              className="sess-rerun"
              disabled={rerunBlocked || locked}
              title={t('sessions.rerunLine')}
              onClick={() => onRerun(line.id)}
            >
              <RotateCw size={13} />
              <span>{isRerunning ? t('sessions.rerunning') : t('sessions.rerunLine')}</span>
            </button>
          </div>
        )}
        {editing ? (
          <textarea
            className="sess-source sess-editing"
            lang={line.lang}
            autoFocus
            rows={Math.max(1, Math.ceil(editing.text.length / 40))}
            value={editing.text}
            onChange={e => onDraft({ id: line.id, text: e.target.value })}
            onBlur={() => onSave(line.id, editing.text, line.source)}
            onKeyDown={e => {
              if (e.key === 'Escape') onDraft(null);
              // Enter saves, shift+Enter breaks the line: a transcript line is one
              // utterance, so the common case is finishing rather than continuing.
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                e.currentTarget.blur();
              }
            }}
          />
        ) : (
          <p
            className={`sess-source${locked ? ' sess-source-locked' : ''}`}
            lang={line.lang}
            title={locked ? t('sessions.editLocked') : t('sessions.editHint')}
            onClick={() => {
              if (!locked) onDraft({ id: line.id, text: line.source });
            }}
          >
            {line.source}
          </p>
        )}
        {langs
          .filter(l => line.translations[l])
          .map(l => (
            <p key={l} className="sess-translation" lang={l}>
              {line.translations[l]}
            </p>
          ))}
        {/* One placeholder, not one per language. Without any, the translations simply
            vanish and read as "this meeting had no Vietnamese" rather than "this line
            failed to translate" — but repeating it per target language (and for the
            line's own language) turns one failure into three lines of noise. */}
        {line.status === 'translate_failed' && (
          <p className="sess-translation sess-translation-missing">{t('sessions.translationMissing')}</p>
        )}
      </div>
    </article>
  );
}

/* 943 rows, and every click used to repaint all of them: `playing` and `draft` were transcript-wide,
   so any change to either invalidated every row. Memoised, with per-row props, a play click repaints
   the row that stopped and the row that started. */
export const TranscriptRow = memo(Row);
