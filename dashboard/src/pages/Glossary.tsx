import { useState, useEffect, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { Loader2, BookMarked, Plus, Trash2 } from 'lucide-react';
import { translateApi, type GlossaryTerm, type PendingGlossaryTerm, type TranslationCandidate, type PhraseCandidate } from '../services/api';
import { useDocumentTitle } from '../hooks/useDocumentTitle';
import { useRole } from '../hooks/useRole';
import { useToast } from '../components/Toast';
import { PageHeader } from '../components/PageHeader';
import { EditableKeyValueTable } from '../components/EditableKeyValueTable';
import { GlossaryPending } from './GlossaryPending';
import { MemoryCandidates } from './MemoryCandidates';
import { pageWindow } from '../utils/pageWindow';
import '../components/EditableTable.css';

const CANDIDATES_PAGE_SIZE = 20;

// 內建類別走 i18n label；自訂類別以字串本身當 value 與 label
const BUILTIN_CATEGORIES = ['name', 'dept', 'term', 'asset', 'phrase', 'other'];

export function Glossary() {
  const { t } = useTranslation();
  useDocumentTitle(t('glossary.title'));
  const { canWrite } = useRole();
  const toast = useToast();

  // Declared before the loader effect that calls it — react-hooks/immutability rejects reading a
  // `const` from above its own declaration, even though the closure only runs after mount.
  const fail = (err: unknown) =>
    toast.error(t('common.failed', { message: err instanceof Error ? err.message : 'unknown' }));

  const [terms, setTerms] = useState<GlossaryTerm[]>([]);
  const [pending, setPending] = useState<PendingGlossaryTerm[]>([]);
  const [candidates, setCandidates] = useState<TranslationCandidate[]>([]);
  const [candTotal, setCandTotal] = useState(0);
  const [candPage, setCandPage] = useState(1);
  const [phrases, setPhrases] = useState<PhraseCandidate[]>([]);
  const [customCategories, setCustomCategories] = useState<string[]>([]);
  const [catInput, setCatInput] = useState('');
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [scanning, setScanning] = useState(false);
  const [tab, setTab] = useState<'candidates' | 'phrases' | 'terms' | 'categories'>('candidates');
  const [showDups, setShowDups] = useState(false);

  // Group by 越南文: source keys are unique, so the only real "duplicate" is many 中文 sharing one target.
  const dupGroups = useMemo(() => {
    const byTarget = new Map<string, GlossaryTerm[]>();
    for (const term of terms) {
      const arr = byTarget.get(term.target) ?? [];
      arr.push(term);
      byTarget.set(term.target, arr);
    }
    return [...byTarget.values()].filter(g => g.length > 1).sort((a, b) => b.length - a.length);
  }, [terms]);

  useEffect(() => {
    let active = true;
    translateApi
      .getGlossary()
      .then(list => active && setTerms(list))
      .catch(err => active && fail(err))
      .finally(() => active && setLoading(false));
    translateApi
      .getPendingGlossary()
      .then(list => active && setPending(list))
      .catch(err => active && fail(err));
    translateApi
      .getMemoryCandidates(CANDIDATES_PAGE_SIZE, 0)
      .then(res => {
        if (!active) return;
        setCandidates(res.items);
        setCandTotal(res.total);
      })
      .catch(err => active && fail(err));
    translateApi
      .getPhraseCandidates()
      .then(list => active && setPhrases(list))
      .catch(err => active && fail(err));
    translateApi
      .getCategories()
      .then(list => active && setCustomCategories(list.filter(c => !BUILTIN_CATEGORIES.includes(c))))
      .catch(err => active && fail(err));
    return () => {
      active = false;
    };
  }, []);

  const scanPhrases = async () => {
    setScanning(true);
    try {
      setPhrases(await translateApi.scanPhraseCandidates());
    } catch (err) {
      fail(err);
    } finally {
      setScanning(false);
    }
  };

  const approvePhrase = async (id: number) => {
    setBusy(true);
    try {
      setPhrases(await translateApi.approvePhraseCandidate(id));
      setTerms(await translateApi.getGlossary());
      toast.success(t('glossary.candidateApproved'));
    } catch (err) {
      fail(err);
    } finally {
      setBusy(false);
    }
  };

  const dismissPhrase = async (id: number) => {
    setBusy(true);
    try {
      setPhrases(await translateApi.dismissPhraseCandidate(id));
    } catch (err) {
      fail(err);
    } finally {
      setBusy(false);
    }
  };

  // Reload one candidate page. If a mutation empties the last page, step back
  // one so the user never lands on a blank page.
  const loadCandidates = async (page: number) => {
    const res = await translateApi.getMemoryCandidates(CANDIDATES_PAGE_SIZE, (page - 1) * CANDIDATES_PAGE_SIZE);
    if (res.items.length === 0 && page > 1) return loadCandidates(page - 1);
    setCandidates(res.items);
    setCandTotal(res.total);
    setCandPage(page);
  };

  const approveCandidate = async (id: number) => {
    setBusy(true);
    try {
      await translateApi.approveMemoryCandidate(id);
      setTerms(await translateApi.getGlossary());
      await loadCandidates(candPage);
      toast.success(t('glossary.candidateApproved'));
    } catch (err) {
      fail(err);
    } finally {
      setBusy(false);
    }
  };

  const dismissCandidate = async (id: number) => {
    setBusy(true);
    try {
      await translateApi.dismissMemoryCandidate(id);
      await loadCandidates(candPage);
    } catch (err) {
      fail(err);
    } finally {
      setBusy(false);
    }
  };

  const add = async (zh: string, vi: string, category?: string) => {
    setBusy(true);
    try {
      setTerms(await translateApi.addGlossaryTerm(zh, vi, category));
      toast.success(t('glossary.added'));
      return true;
    } catch (err) {
      fail(err);
      return false;
    } finally {
      setBusy(false);
    }
  };

  const remove = async (term: string) => {
    setBusy(true);
    try {
      setTerms(await translateApi.removeGlossaryTerm(term));
    } catch (err) {
      fail(err);
    } finally {
      setBusy(false);
    }
  };

  const saveEdit = async (original: string, zh: string, vi: string, category?: string) => {
    setBusy(true);
    try {
      // POST upserts on the source key, so an unchanged source is a plain overwrite. A changed
      // one writes a new record, which leaves the old key behind until it is removed.
      let list = await translateApi.addGlossaryTerm(zh, vi, category);
      if (zh !== original) list = await translateApi.removeGlossaryTerm(original);
      setTerms(list);
      toast.success(t('common.saved'));
      return true;
    } catch (err) {
      fail(err);
      return false;
    } finally {
      setBusy(false);
    }
  };

  const addCat = async () => {
    const name = catInput.trim();
    if (!name) return;
    setBusy(true);
    try {
      const list = await translateApi.addCategory(name);
      setCustomCategories(list.filter(c => !BUILTIN_CATEGORIES.includes(c)));
      setCatInput('');
    } catch (err) {
      fail(err);
    } finally {
      setBusy(false);
    }
  };

  const removeCat = async (name: string) => {
    setBusy(true);
    try {
      const list = await translateApi.deleteCategory(name);
      setCustomCategories(list.filter(c => !BUILTIN_CATEGORIES.includes(c)));
    } catch (err) {
      fail(err);
    } finally {
      setBusy(false);
    }
  };

  const categoryOptions = [
    { value: '', label: t('glossary.category.unset') },
    { value: 'name', label: t('glossary.category.name') },
    { value: 'dept', label: t('glossary.category.dept') },
    { value: 'term', label: t('glossary.category.term') },
    { value: 'asset', label: t('glossary.category.asset') },
    { value: 'phrase', label: t('glossary.category.phrase') },
    { value: 'other', label: t('glossary.category.other') },
    ...customCategories.map(c => ({ value: c, label: c })),
  ];

  const refetch = async () => {
    const [list, pend] = await Promise.all([
      translateApi.getGlossary(),
      translateApi.getPendingGlossary(),
    ]);
    setTerms(list);
    setPending(pend);
  };

  const approve = async (id: number) => {
    setBusy(true);
    try {
      await translateApi.approvePendingGlossary(id);
      await refetch();
      toast.success(t('glossary.approved'));
    } catch (err) {
      fail(err);
    } finally {
      setBusy(false);
    }
  };

  const reject = async (id: number) => {
    setBusy(true);
    try {
      await translateApi.rejectPendingGlossary(id);
      setPending(await translateApi.getPendingGlossary());
      toast.success(t('glossary.rejected'));
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
      <PageHeader
        title={t('glossary.title')}
        subtitle={t('glossary.subtitle')}
      />

      <div className="etable-tabs" role="tablist">
        <button
          role="tab"
          aria-selected={tab === 'candidates'}
          className={`etable-tab ${tab === 'candidates' ? 'active' : ''}`}
          onClick={() => setTab('candidates')}
        >
          {t('glossary.candidatesTitle')}
          <span className="etable-count">{candTotal}</span>
        </button>
        <button
          role="tab"
          aria-selected={tab === 'phrases'}
          className={`etable-tab ${tab === 'phrases' ? 'active' : ''}`}
          onClick={() => setTab('phrases')}
        >
          {t('glossary.phrasesTitle')}
          <span className="etable-count">{phrases.length}</span>
        </button>
        <button
          role="tab"
          aria-selected={tab === 'terms'}
          className={`etable-tab ${tab === 'terms' ? 'active' : ''}`}
          onClick={() => setTab('terms')}
        >
          {t('glossary.terms')}
          <span className="etable-count">{terms.length}</span>
        </button>
        <button
          role="tab"
          aria-selected={tab === 'categories'}
          className={`etable-tab ${tab === 'categories' ? 'active' : ''}`}
          onClick={() => setTab('categories')}
        >
          {t('glossary.category.manageTitle')}
          <span className="etable-count">{customCategories.length}</span>
        </button>
      </div>

      {tab === 'candidates' && (
        <>
          <MemoryCandidates
            candidates={candidates}
            canWrite={canWrite}
            busy={busy}
            onApprove={approveCandidate}
            onDismiss={dismissCandidate}
          />
          {candTotal > CANDIDATES_PAGE_SIZE && (
            <div className="pagination">
              <button disabled={candPage === 1 || busy} onClick={() => loadCandidates(candPage - 1)}>
                {t('common.previous')}
              </button>
              <span className="page-numbers">
                {pageWindow(candPage, Math.ceil(candTotal / CANDIDATES_PAGE_SIZE)).map(p => (
                  <button
                    key={p}
                    className={p === candPage ? 'active' : ''}
                    disabled={busy}
                    onClick={() => loadCandidates(p)}
                  >
                    {p}
                  </button>
                ))}
              </span>
              <button
                disabled={candPage >= Math.ceil(candTotal / CANDIDATES_PAGE_SIZE) || busy}
                onClick={() => loadCandidates(candPage + 1)}
              >
                {t('common.next')}
              </button>
            </div>
          )}
        </>
      )}

      {tab === 'phrases' && (
        <>
          <section className="etable-panel">
            {/* No panel title: the glossary tab bar already shows this count. */}
            <p className="etable-empty">{t('glossary.phrasesHint')}</p>
            {canWrite && (
              <button className="etable-add" onClick={scanPhrases} disabled={scanning}>
                {scanning ? <Loader2 className="animate-spin" size={16} /> : null}
                {t('glossary.phrasesScan')}
              </button>
            )}
          </section>
          {/* Reuse the memory-candidate row UI: phrase maps onto the source column. */}
          <MemoryCandidates
            candidates={phrases.map(p => ({ id: p.id, pairKey: '', source: p.phrase, translated: p.translated, count: p.count, at: p.at }))}
            canWrite={canWrite}
            busy={busy}
            onApprove={approvePhrase}
            onDismiss={dismissPhrase}
          />
        </>
      )}

      {tab === 'terms' && (
        <>
          <GlossaryPending
            pending={pending}
            canWrite={canWrite}
            busy={busy}
            onApprove={approve}
            onReject={reject}
          />

          <section className="etable-panel etable-panel--dup">
            <button className="etable-add" onClick={() => setShowDups(v => !v)}>
              {t('glossary.dupCheck')}
              <span className="etable-count">{dupGroups.length}</span>
            </button>
            {showDups &&
              (dupGroups.length === 0 ? (
                <div className="etable-empty">{t('glossary.dupEmpty')}</div>
              ) : (
                <>
                  <h3 className="etable-panel-title">{t('glossary.dupTitle')}</h3>
                  <div className="etable-list">
                    {dupGroups.map(group => (
                      <div key={group[0].target} className="etable-item etable-item--simple">
                        <span className="etable-val">{group[0].target}</span>
                        <div className="etable-row-actions etable-dup-terms">
                          {group.map(g => (
                            <span key={g.source} className="etable-src">
                              {g.source}
                              {canWrite && (
                                <button
                                  className="etable-del"
                                  onClick={() => remove(g.source)}
                                  disabled={busy}
                                  title={t('glossary.category.deleteButton')}
                                >
                                  <Trash2 size={16} />
                                </button>
                              )}
                            </span>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </>
              ))}
          </section>

          {/* The glossary maps 中文 to Tiếng Việt, so the key/val labels name the languages themselves
              and are written in their own script, as a language picker would. */}
          <EditableKeyValueTable
            rows={terms}
            titleLabel={t('glossary.terms')}
            hideTitle
            keyLabel="中文"
            valLabel="Tiếng Việt"
            addLabel={t('glossary.add')}
            emptyIcon={<BookMarked size={32} strokeWidth={1} />}
            emptyText={t('glossary.empty')}
            canWrite={canWrite}
            busy={busy}
            resizeStorageKey="glossary-col-src"
            initialSortKey="key"
            rowKey={g => g.source}
            rowVal={g => g.target}
            rowCount={g => g.count ?? 0}
            compareKey={(a, b) => a.source.localeCompare(b.source)}
            compareVal={(a, b) => a.target.localeCompare(b.target)}
            tieBreak={(a, b) => a.source.localeCompare(b.source)}
            categoryLabel={t('glossary.category.label')}
            categoryOptions={categoryOptions}
            rowCategory={g => g.category ?? ''}
            onAdd={add}
            onSaveEdit={saveEdit}
            onRemove={remove}
          />
        </>
      )}

      {tab === 'categories' && (
        <section className="etable-panel">
          <h3 className="etable-panel-title">{t('glossary.category.manageTitle')}</h3>
          {canWrite && (
            <div className="etable-add">
              <input
                value={catInput}
                onChange={e => setCatInput(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && addCat()}
                placeholder={t('glossary.category.addPlaceholder')}
              />
              <button className="btn-primary" onClick={addCat} disabled={busy || !catInput.trim()}>
                <Plus size={16} />
                {t('glossary.category.addButton')}
              </button>
            </div>
          )}
          <div className="etable-list">
            {customCategories.length === 0 ? (
              <div className="etable-empty">{t('glossary.category.empty')}</div>
            ) : (
              customCategories.map(c => (
                <div key={c} className="etable-item etable-item--simple">
                  <span className="etable-src">{c}</span>
                  {canWrite && (
                    <div className="etable-row-actions">
                      <button
                        className="etable-del"
                        onClick={() => removeCat(c)}
                        disabled={busy}
                        title={t('glossary.category.deleteButton')}
                      >
                        <Trash2 size={16} />
                      </button>
                    </div>
                  )}
                </div>
              ))
            )}
          </div>
        </section>
      )}
    </div>
  );
}

export default Glossary;
