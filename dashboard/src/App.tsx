import { Suspense } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { lazyWithRetry as lazy } from './utils/lazyWithRetry';
import { Loader2 } from 'lucide-react';
import { Layout } from './components/Layout';
import { ToastProvider } from './components/ToastProvider';
import { ErrorBoundary } from './components/ErrorBoundary';
import './App.css';

const Glossary = lazy(() => import('./pages/Glossary').then(m => ({ default: m.Glossary })));
const LlmSettings = lazy(() => import('./pages/LlmSettings').then(m => ({ default: m.LlmSettings })));
const KeyProxy = lazy(() => import('./pages/KeyProxy').then(m => ({ default: m.KeyProxy })));
const Settings = lazy(() => import('./pages/Settings').then(m => ({ default: m.Settings })));

const loadingFallback = (
  <div className="app-loading">
    <Loader2 className="animate-spin" size={32} />
  </div>
);

function App() {
  return (
    <ErrorBoundary>
      <ToastProvider>
        <BrowserRouter>
          <Suspense fallback={loadingFallback}>
            <Routes>
              <Route path="/" element={<Layout />}>
                <Route index element={<Navigate to="/glossary" replace />} />
                <Route path="glossary" element={<Glossary />} />
                <Route path="settings" element={<Settings />}>
                  <Route index element={<Navigate to="llm" replace />} />
                  <Route path="llm" element={<LlmSettings />} />
                  <Route path="keyproxy" element={<KeyProxy />} />
                </Route>
                <Route path="*" element={<Navigate to="/glossary" replace />} />
              </Route>
            </Routes>
          </Suspense>
        </BrowserRouter>
      </ToastProvider>
    </ErrorBoundary>
  );
}

export default App;
