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

test('caps the buffer to max, dropping the oldest line off the front', () => {
  let lines = Array.from({ length: 200 }, (_, i) => line(i + 1, i));
  lines = mergeLine(lines, line(201, 200), 200);
  assert.equal(lines.length, 200, 'buffer must stay at the cap');
  assert.equal(lines[0].id, 2, 'the oldest line is dropped');
  assert.equal(lines[lines.length - 1].id, 201, 'the newest line is kept');
});

test('an in-place revision does not grow the buffer past the cap', () => {
  const lines = Array.from({ length: 200 }, (_, i) => ({ id: i + 1, start: i, refined: false }));
  const revised = mergeLine(lines, { id: 150, start: 149, refined: true }, 200);
  assert.equal(revised.length, 200);
  assert.equal(revised.find(l => l.id === 150)!.refined, true);
});

test('without a max the buffer is unbounded, preserving existing callers', () => {
  let lines = [] as ReturnType<typeof line>[];
  for (let i = 1; i <= 300; i++) lines = mergeLine(lines, line(i, i));
  assert.equal(lines.length, 300);
});

test('does not mutate the array it was given', () => {
  const before = [line(1, 0), line(3, 30)];
  const copy = [...before];
  mergeLine(before, line(2, 15));
  assert.deepEqual(before, copy);
});
