import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Loader2, Save, ExternalLink } from 'lucide-react';
import { PageHeader } from '../components/PageHeader';
import { useToast } from '../components/Toast';
import { useDocumentTitle } from '../hooks/useDocumentTitle';
import { appApi, type AppConfig, type DisplaySettings } from '../services/app.api';
import './Display.css';

const LANGUAGE_CHOICES = ['zh', 'vi', 'en', 'ja', 'ko', 'th', 'id'];

export function Display() {
  const { t } = useTranslation();
  useDocumentTitle(t('display.title'));
  const toast = useToast();

  const [cfg, setCfg] = useState<AppConfig | null>(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    let alive = true;
    appApi
      .getConfig()
      .then(c => alive && setCfg(c))
      .catch(err => alive && toast.error(String(err)));
    return () => {
      alive = false;
    };
  }, []);

  if (!cfg) {
    return (
      <div className="etable-page etable-loading">
        <Loader2 className="animate-spin" size={32} />
      </div>
    );
  }

  const patchDisplay = (patch: Partial<DisplaySettings>) =>
    setCfg({ ...cfg, display: { ...cfg.display, ...patch } });

  const setLanguage = (index: number, code: string) => {
    const next = [...cfg.languages];
    if (code === '') next.splice(index, 1);
    else next[index] = code;
    setCfg({ ...cfg, languages: next });
  };

  const save = async () => {
    setSaving(true);
    try {
      setCfg(await appApi.putConfig({ languages: cfg.languages, display: cfg.display }));
      toast.success(t('common.saved'));
    } catch (err) {
      toast.error(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  };

  const used = new Set(cfg.languages);

  return (
    <div className="etable-page">
      <PageHeader
        title={t('display.title')}
        subtitle={t('display.subtitle')}
        actions={
          <button className="btn-primary" onClick={save} disabled={saving}>
            {saving ? <Loader2 size={18} className="animate-spin" /> : <Save size={18} />}
            {t('common.save')}
          </button>
        }
      />

      <section className="etable-panel">
        <h3 className="etable-panel-title">{t('display.languages')}</h3>
        <p className="display-hint">{t('display.languagesHint')}</p>
        <div className="display-langs">
          {[0, 1, 2].map(i => (
            <label key={i} className="display-field">
              <span>{i === 0 ? t('display.primary') : t('display.language', { n: i + 1 })}</span>
              <select value={cfg.languages[i] ?? ''} onChange={e => setLanguage(i, e.target.value)}>
                {i > 1 && <option value="">{t('display.none')}</option>}
                {LANGUAGE_CHOICES.filter(c => c === cfg.languages[i] || !used.has(c)).map(c => (
                  <option key={c} value={c}>
                    {t(`lang.${c}`, { defaultValue: c })}
                  </option>
                ))}
              </select>
            </label>
          ))}
        </div>
      </section>

      <section className="etable-panel">
        <h3 className="etable-panel-title">{t('display.format')}</h3>
        <div className="display-grid">
          <label className="display-field">
            <span>{t('display.fontSize')}</span>
            <input
              type="range"
              min={20}
              max={90}
              value={cfg.display.font_size}
              onChange={e => patchDisplay({ font_size: Number(e.target.value) })}
            />
            <output>{cfg.display.font_size}px</output>
          </label>

          <label className="display-field">
            <span>{t('display.lines')}</span>
            <input
              type="number"
              min={1}
              max={20}
              value={cfg.display.lines}
              onChange={e => patchDisplay({ lines: Number(e.target.value) })}
            />
          </label>

          <label className="display-field">
            <span>{t('display.source')}</span>
            <select
              value={cfg.display.show_source}
              onChange={e => patchDisplay({ show_source: e.target.value as DisplaySettings['show_source'] })}
            >
              <option value="top">{t('display.sourceTop')}</option>
              <option value="bottom">{t('display.sourceBottom')}</option>
              <option value="hidden">{t('display.sourceHidden')}</option>
            </select>
          </label>

          <label className="display-field">
            <span>{t('display.theme')}</span>
            <select
              value={cfg.display.theme}
              onChange={e => patchDisplay({ theme: e.target.value as DisplaySettings['theme'] })}
            >
              <option value="dark">{t('display.themeDark')}</option>
              <option value="light">{t('display.themeLight')}</option>
            </select>
          </label>

          <label className="display-check">
            <input
              type="checkbox"
              checked={cfg.display.show_speaker}
              onChange={e => patchDisplay({ show_speaker: e.target.checked })}
            />
            <span>{t('display.showSpeaker')}</span>
          </label>

          <label className="display-check">
            <input
              type="checkbox"
              checked={cfg.display.colour_speakers}
              onChange={e => patchDisplay({ colour_speakers: e.target.checked })}
            />
            <span>{t('display.colourSpeakers')}</span>
          </label>
        </div>

        <a className="display-open" href="/live" target="_blank" rel="noreferrer">
          <ExternalLink size={16} />
          {t('display.openLive')}
        </a>
      </section>
    </div>
  );
}

export default Display;
