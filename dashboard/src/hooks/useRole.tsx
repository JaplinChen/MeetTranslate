// Single-user local app: no login, no roles. Kept as a hook because the pages copied from
// OpenWA-Lab gate their edit controls on canWrite — one constant here beats editing every page.
export function useRole(): { canWrite: boolean } {
  return { canWrite: true };
}
