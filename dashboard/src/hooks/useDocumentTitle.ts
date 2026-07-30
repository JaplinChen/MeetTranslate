import { useEffect } from 'react';

/**
 * Custom hook to set document title dynamically.
 * Automatically appends " | MeetTranslate" suffix.
 */
export function useDocumentTitle(title: string) {
  useEffect(() => {
    const previousTitle = document.title;
    document.title = `${title} | MeetTranslate`;

    return () => {
      document.title = previousTitle;
    };
  }, [title]);
}
