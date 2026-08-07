import { test } from 'node:test';
import assert from 'node:assert/strict';

import { restoreColumnWidths } from './useResizableCol.ts';

function fakeEl() {
  const set: Array<[string, string]> = [];
  return { set, el: { style: { setProperty: (n: string, v: string) => set.push([n, v]) } } as unknown as HTMLElement };
}

test('applies every saved width as a --col- custom property', () => {
  const { set, el } = fakeEl();
  restoreColumnWidths(el, JSON.stringify({ provider: '10rem', key: '240px' }));
  assert.deepEqual(set, [['--col-provider', '10rem'], ['--col-key', '240px']]);
});

test('does nothing when there is no saved entry', () => {
  const { set, el } = fakeEl();
  restoreColumnWidths(el, null);
  assert.equal(set.length, 0);
});

test('ignores a corrupt entry rather than throwing', () => {
  const { set, el } = fakeEl();
  restoreColumnWidths(el, '{not json');
  assert.equal(set.length, 0);
});
