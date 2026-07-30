import { Loader2 } from 'lucide-react';

interface PageLoaderProps {
  className?: string;
}

export function PageLoader({ className }: PageLoaderProps) {
  return (
    <div className={className ? `${className} page-loading` : 'page-loading'}>
      <Loader2 className="animate-spin" size={32} />
    </div>
  );
}
