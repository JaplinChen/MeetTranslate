import { Suspense } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { lazyWithRetry as lazy } from './utils/lazyWithRetry';
import { Loader2 } from 'lucide-react';
import { Layout } from './components/Layout';
import { ToastProvider } from './components/ToastProvider';
import { ErrorBoundary } from './components/ErrorBoundary';
import './App.css';

const Capture = lazy(() => import('./pages/Capture').then(m => ({ default: m.Capture })));
const Sessions = lazy(() => import('./pages/Sessions').then(m => ({ default: m.Sessions })));
const Glossary = lazy(() => import('./pages/Glossary').then(m => ({ default: m.Glossary })));
const Live = lazy(() => import('./pages/Live').then(m => ({ default: m.Live })));
const Display = lazy(() => import('./pages/Display').then(m => ({ default: m.Display })));
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
              {/* Outside the dashboard shell: this one goes fullscreen on the meeting-room TV. */}
              <Route path="/live" element={<Live />} />
              <Route path="/" element={<Layout />}>
                <Route index element={<Navigate to="/capture" replace />} />
                <Route path="capture" element={<Capture />} />
                <Route path="sessions" element={<Sessions />} />
                <Route path="glossary" element={<Glossary />} />
                <Route path="settings" element={<Settings />}>
                  <Route index element={<Navigate to="display" replace />} />
                  <Route path="display" element={<Display />} />
                  <Route path="llm" element={<LlmSettings />} />
                  <Route path="keyproxy" element={<KeyProxy />} />
                </Route>
                <Route path="*" element={<Navigate to="/capture" replace />} />
              </Route>
            </Routes>
          </Suspense>
        </BrowserRouter>
      </ToastProvider>
    </ErrorBoundary>
  );
}

export default App;
