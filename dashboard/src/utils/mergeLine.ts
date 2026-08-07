/** One subtitle line as the socket delivers it. Only the fields ordering depends on. */
export interface OrderedLine {
  id: number;
  start: number;
}

/**
 * Fold one incoming line into the list on screen.
 *
 * A known id replaces in place: the server revises a line once it has seen what came next, and
 * appending that would show the same sentence twice.
 *
 * A new id is inserted by start time rather than appended, because lines do not always arrive in
 * order. An utterance the recogniser gave up on is held and retried once its speaker's language is
 * settled, so it arrives after lines from later in the meeting. Appended, it would sit at the
 * bottom of the meeting-room TV and then jump into place on the next reload, since the stored
 * transcript is ordered by start.
 */
export function mergeLine<T extends OrderedLine>(previous: T[], line: T, max = Infinity): T[] {
  const known = previous.findIndex(l => l.id === line.id);
  let next: T[];
  if (known !== -1) {
    next = [...previous];
    next[known] = line;
  } else {
    const at = previous.findIndex(l => l.start > line.start);
    next = at === -1 ? [...previous, line] : [...previous.slice(0, at), line, ...previous.slice(at)];
  }

  // Cap the buffer: only the last `display.lines` are ever shown, so leaving this unbounded grew
  // the array for the whole meeting — thousands of lines on a multi-hour call, an O(n) findIndex on
  // every message and memory that only climbs. The oldest are off-screen, so trim from the front.
  return next.length > max ? next.slice(next.length - max) : next;
}
