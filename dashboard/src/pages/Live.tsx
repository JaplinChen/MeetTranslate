import { useEffect, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useDocumentTitle } from '../hooks/useDocumentTitle';
import { useLiveSocket, type LiveLine, type DisplaySettings } from '../hooks/useLiveSocket';
import './Live.css';

// Distinct hues rather than a palette lookup: the number of speakers is not known ahead of time,
// and evenly spaced hues stay distinguishable at TV viewing distance for any count.
const speakerHue = (code: string): number => {
  const n = Number.parseInt(code.replace(/\D/g, ''), 10);
  return Number.isFinite(n) ? (n * 137) % 360 : 0;
};

function Row({ line, display, languages }: { line: LiveLine; display: DisplaySettings; languages: string[] }) {
  const targets = languages.filter(l => l !== line.lang && line.translations[l]);
  const hue = speakerHue(line.speaker);

  const source = display.show_source !== 'hidden' && (
    <p className="live-source" lang={line.lang}>
      {line.source}
    </p>
  );

  return (
    <article
      className={`live-row${line.refined ? ' is-refined' : ''}`}
      style={display.colour_speakers ? ({ '--speaker-hue': hue } as React.CSSProperties) : undefined}
    >
      {display.show_speaker && <span className="live-speaker">{line.speaker}</span>}
      <div className="live-text">
        {display.show_source === 'top' && source}
        {targets.map(lang => (
          <p key={lang} className="live-translation" lang={lang}>
            {line.translations[lang]}
          </p>
        ))}
        {display.show_source === 'bottom' && source}
      </div>
    </article>
  );
}

export function Live() {
  const { t } = useTranslation();
  useDocumentTitle(t('live.title'));
  const { lines, display, languages, connected } = useLiveSocket();
  const [atBottom, setAtBottom] = useState(true);
  const endRef = useRef<HTMLDivElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  const visible = useMemo(() => lines.slice(-display.lines), [lines, display.lines]);

  // Only auto-scroll when the viewer has not scrolled up to read something older.
  useEffect(() => {
    if (atBottom) endRef.current?.scrollIntoView({ block: 'end' });
  }, [visible, atBottom]);

  const onScroll = () => {
    const el = scrollRef.current;
    if (el) setAtBottom(el.scrollHeight - el.scrollTop - el.clientHeight < 40);
  };

  return (
    <div
      className={`live-page live-theme-${display.theme}`}
      style={{ '--live-font': `${display.font_size}px` } as React.CSSProperties}
    >
      {!connected && <div className="live-banner">{t('live.connecting')}</div>}
      <div className="live-scroll" ref={scrollRef} onScroll={onScroll}>
        {visible.length === 0 ? (
          <p className="live-idle">{t('live.waiting')}</p>
        ) : (
          visible.map(line => <Row key={line.id} line={line} display={display} languages={languages} />)
        )}
        <div ref={endRef} />
      </div>
    </div>
  );
}

export default Live;
