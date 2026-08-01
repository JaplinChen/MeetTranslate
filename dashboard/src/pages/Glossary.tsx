import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { BookMarked, Loader2, Plus, Trash2 } from 'lucide-react';
import { PageHeader } from '../components/PageHeader';
import { useToast } from '../components/Toast';
import { useDocumentTitle } from '../hooks/useDocumentTitle';
import { appApi, type GlossaryTerm } from '../services/app.api';
import './Glossary.css';

const MODES = ['translate', 'keep', 'hint'] as const;

const emptyDraft = () => ({
  source: '',
  mode: 'translate' as GlossaryTerm['mode'],
  category: '',
  targets: {} as Record<string, string>,
});

export function Glossary() {
  const { t } = useTranslation();
  useDocumentTitle(t('glossary.title'));
  const toast = useToast();

  const [terms, setTerms] = useState<GlossaryTerm[]>([]);
  const [languages, setLanguages] = useState<string[]>([]);
  const [draft, setDraft] = useState(emptyDraft());
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [clash, setClash] = useState<{ text: string; count: number }[]>([]);

  const fail = (err: unknown) => toast.error(err instanceof Error ? err.message : String(err));

  useEffect(() => {
    let alive = true;
    Promise.all([appApi.glossary(), appApi.getConfig()])
      .then(([list, cfg]) => {
        if (!alive) return;
        setTerms(list);
        setLanguages(cfg.languages);
      })
      .catch(err => alive && fail(err))
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
  }, []);

  const add = async () => {
    if (!draft.source.trim()) return;
    setBusy(true);
    try {
      setTerms(await appApi.addTerm(draft));
      setDraft(emptyDraft());
    } catch (err) {
      fail(err);
    } finally {
      setBusy(false);
    }
  };

  // The corrector rewrites anything whose pinyin matches a term, and Mandarin supplies
  // homophones for almost everything. Checked against the meetings already recorded, so the
  // answer is about what these people say rather than what the language permits.
  const checkClash = async (source: string) => {
    if (!source.trim()) return setClash([]);
    try {
      setClash((await appApi.termCollisions(source.trim())).collisions);
    } catch {
      setClash([]);
    }
  };

  const remove = async (term: GlossaryTerm) => {
    setBusy(true);
    try {
      setTerms(await appApi.removeTerm(term.source, term.lang));
    } catch (err) {
      fail(err);
    } finally {
      setBusy(false);
    }
  };

  if (loading) {
    return (
      <div className="etable-page etable-loading">
        <Loader2 className="animate-spin" size={32} />
      </div>
    );
  }

  return (
    <div className="etable-page">
      <PageHeader title={t('glossary.title')} subtitle={t('glossary.subtitle')} />

      <section className="etable-panel">
        {clash.length > 0 && (
          <p className="gloss-clash">
            {t('glossary.clashWarning', {
              list: clash.map(c => `${c.text} (${c.count})`).join('、'),
            })}
          </p>
        )}
        <div className="gloss-add">
          <input
            className="gloss-input"
            placeholder={t('glossary.sourcePlaceholder')}
            value={draft.source}
            onChange={e => setDraft({ ...draft, source: e.target.value })}
            onBlur={e => checkClash(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && !busy && add()}
          />
          <select
            value={draft.mode}
            onChange={e => setDraft({ ...draft, mode: e.target.value as GlossaryTerm['mode'] })}
          >
            {MODES.map(m => (
              <option key={m} value={m}>
                {t(`glossary.mode.${m}`)}
              </option>
            ))}
          </select>
          {/* Only `translate` needs per-language wording — `keep` and `hint` have no target text. */}
          {draft.mode === 'translate' &&
            languages.map(lang => (
              <input
                key={lang}
                className="gloss-input gloss-target"
                placeholder={t(`lang.${lang}`, { defaultValue: lang })}
                value={draft.targets[lang] ?? ''}
                onChange={e => setDraft({ ...draft, targets: { ...draft.targets, [lang]: e.target.value } })}
              />
            ))}
          <input
            className="gloss-input gloss-category"
            placeholder={t('glossary.categoryPlaceholder')}
            value={draft.category}
            onChange={e => setDraft({ ...draft, category: e.target.value })}
          />
          <button className="btn-primary" onClick={add} disabled={busy || !draft.source.trim()}>
            <Plus size={16} />
            {t('glossary.add')}
          </button>
        </div>
        <p className="gloss-hint">{t('glossary.modeHint')}</p>
      </section>

      <section className="etable-panel">
        <div className="etable-panel-title">
          {t('glossary.terms')}
          <span className="etable-count">{terms.length}</span>
        </div>
        {terms.length === 0 ? (
          <div className="gloss-empty">
            <BookMarked size={32} strokeWidth={1} />
            <span>{t('glossary.empty')}</span>
          </div>
        ) : (
          <div className="gloss-scroll">
            <table className="gloss-table">
              <thead>
                <tr>
                  <th>{t('glossary.source')}</th>
                  <th>{t('glossary.modeLabel')}</th>
                  {languages.map(l => (
                    <th key={l}>{t(`lang.${l}`, { defaultValue: l })}</th>
                  ))}
                  <th>{t('glossary.category')}</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {terms.map(term => (
                  <tr key={`${term.source}/${term.lang}`}>
                    <td className="gloss-src">{term.source}</td>
                    <td>
                      <span className={`gloss-badge gloss-badge-${term.mode}`}>{t(`glossary.mode.${term.mode}`)}</span>
                    </td>
                    {languages.map(l => (
                      <td key={l}>{term.mode === 'translate' ? (term.targets[l] ?? '—') : '—'}</td>
                    ))}
                    <td className="gloss-muted">{term.category || '—'}</td>
                    <td>
                      <button
                        className="gloss-del"
                        onClick={() => remove(term)}
                        disabled={busy}
                        title={t('glossary.remove')}
                      >
                        <Trash2 size={16} />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}

export default Glossary;
