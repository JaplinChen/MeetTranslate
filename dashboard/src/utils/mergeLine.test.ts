import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mergeLine } from './mergeLine.ts';

const line = (id: number, start: number) => ({ id, start });

test('appends when lines arrive in order, which is the ordinary case', () => {
  let lines = [] as ReturnType<typeof line>[];
  lines = mergeLine(lines, line(1, 0));
  lines = mergeLine(lines, line(2, 5));
  lines = mergeLine(lines, line(3, 9));
  assert.deepEqual(lines.map(l => l.id), [1, 2, 3]);
});

test('a recovered utterance lands at its own time, not at the bottom', () => {
  // Line 1 failed to decode and was held. Lines 2 and 3 are already on the TV when the retry
  // succeeds, so line 4 carries an earlier start than both.
  let lines = [line(2, 20), line(3, 30)];
  lines = mergeLine(lines, line(4, 10));
  assert.deepEqual(lines.map(l => l.id), [4, 2, 3]);
  assert.deepEqual(lines.map(l => l.start), [10, 20, 30]);
});

test('inserts between two existing lines', () => {
  const lines = mergeLine([line(1, 0), line(3, 30)], line(2, 15));
  assert.deepEqual(lines.map(l => l.start), [0, 15, 30]);
});

test('a revision replaces in place instead of showing the sentence twice', () => {
  const revised = { id: 2, start: 20, refined: true };
  const lines = mergeLine([{ id: 1, start: 0, refined: false }, { id: 2, start: 20, refined: false }], revised);
  assert.equal(lines.length, 2);
  assert.equal(lines[1].refined, true);
  assert.deepEqual(lines.map(l => l.id), [1, 2]);
});

test('a revision keeps its position even if its start changed', () => {
  const lines = mergeLine([line(1, 0), line(2, 20)], { id: 1, start: 99 });
  assert.deepEqual(lines.map(l => l.id), [1, 2], 'replacing must not reorder');
});

test('ties keep the line already on screen first', () => {
  const lines = mergeLine([line(1, 10)], line(2, 10));
  assert.deepEqual(lines.map(l => l.id), [1, 2]);
});

test('does not mutate the array it was given', () => {
  const before = [line(1, 0), line(3, 30)];
  const copy = [...before];
  mergeLine(before, line(2, 15));
  assert.deepEqual(before, copy);
});
