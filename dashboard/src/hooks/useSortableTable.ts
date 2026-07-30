import { useMemo, useState } from 'react';

const PAGE_SIZE = 50;

interface Options<T, K extends string> {
  rows: T[];
  initialKey: K;
  descFirstKeys?: K[];
  searchText: (row: T) => string;
  compare: (a: T, b: T, key: K) => number;
  tieBreak?: (a: T, b: T) => number;
}

export function useSortableTable<T, K extends string>({
  rows,
  initialKey,
  descFirstKeys = [],
  searchText,
  compare,
  tieBreak,
}: Options<T, K>) {
  const [filter, setFilter] = useState('');
  const [sortKey, setSortKey] = useState<K>(initialKey);
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('asc');
  const [page, setPage] = useState(1);

  const filtered = useMemo(() => {
    const q = filter.trim().toLowerCase();
    const list = q ? rows.filter(r => searchText(r).toLowerCase().includes(q)) : [...rows];
    const dir = sortDir === 'asc' ? 1 : -1;
    return list.sort((a, b) => compare(a, b, sortKey) * dir || (tieBreak ? tieBreak(a, b) : 0));
    // searchText/compare/tieBreak are inline closures; memoizing on them would defeat the memo.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rows, filter, sortKey, sortDir]);

  const toggleSort = (key: K) => {
    if (key === sortKey) {
      setSortDir(d => (d === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortKey(key);
      setSortDir(descFirstKeys.includes(key) ? 'desc' : 'asc');
    }
  };

  const sortMark = (key: K) => (sortKey === key ? (sortDir === 'asc' ? ' ▲' : ' ▼') : '');

  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  // Clamp instead of resetting via effect: a filter change that shrinks the list lands on the
  // nearest valid page without an extra render.
  const current = Math.min(page, totalPages);
  const paged = filtered.slice((current - 1) * PAGE_SIZE, current * PAGE_SIZE);

  return { filter, setFilter, filtered, toggleSort, sortMark, current, totalPages, paged, setPage };
}
