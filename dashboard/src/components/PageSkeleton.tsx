import './PageSkeleton.css';

interface PageSkeletonProps {
  rows?: number;
}

/* Placeholder for the four list pages while their first fetch is in flight. It mirrors the real
   layout — header, panel, rows — so the content lands in place instead of replacing a centred
   spinner, which shifted everything on arrival. */
export function PageSkeleton({ rows = 6 }: PageSkeletonProps) {
  return (
    <div className="etable-page skeleton-page" aria-busy="true" aria-live="polite">
      <div className="page-header">
        <div className="page-header__title-group">
          <span className="skeleton-bar skeleton-bar--title" />
        </div>
        <p className="page-header__subtitle">
          <span className="skeleton-bar skeleton-bar--subtitle" />
        </p>
      </div>

      <section className="etable-panel">
        <span className="skeleton-bar skeleton-bar--panel-title" />
        <div className="skeleton-rows">
          {Array.from({ length: rows }, (_, i) => (
            <span key={i} className="skeleton-bar skeleton-bar--row" />
          ))}
        </div>
      </section>
    </div>
  );
}
