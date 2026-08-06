import { Suspense } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { lazyWithRetry as lazy } from './utils/lazyWithRetry';
import { Loader2 } from 'lucide-react';
import { Layout } from './components/Layout';
import { ToastProvider } from './components/ToastProvider';
import { ErrorBoundary } from './components/ErrorBoundary';
import './theme.css';
import './App.css';
/* The shared shell for the list pages — .etable-page, .etable-panel, .etable-item and the rest.
   Six pages and PageSkeleton render those classes, so it loads once here rather than being
   imported (and duplicated) by each of them. There is no EditableTable component and never was;
   the classes are named for the shape the pages share. */
import './components/etable/shell.css';
import './components/etable/table.css';
import './components/etable/controls.css';

const Capture = lazy('Capture', () => import('./pages/Capture').then(m => ({ default: m.Capture })));
const Sessions = lazy('Sessions', () => import('./pages/Sessions').then(m => ({ default: m.Sessions })));
const Glossary = lazy('Glossary', () => import('./pages/Glossary').then(m => ({ default: m.Glossary })));
const Ask = lazy('Ask', () => import('./pages/Ask').then(m => ({ default: m.Ask })));
const Learned = lazy('Learned', () => import('./pages/Learned').then(m => ({ default: m.Learned })));
const Live = lazy('Live', () => import('./pages/Live').then(m => ({ default: m.Live })));
const Display = lazy('Display', () => import('./pages/Display').then(m => ({ default: m.Display })));
const LlmSettings = lazy('LlmSettings', () => import('./pages/LlmSettings').then(m => ({ default: m.LlmSettings })));
const KeyProxy = lazy('KeyProxy', () => import('./pages/KeyProxy').then(m => ({ default: m.KeyProxy })));
const Settings = lazy('Settings', () => import('./pages/Settings').then(m => ({ default: m.Settings })));

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
                <Route path="ask" element={<Ask />} />
                <Route path="glossary" element={<Glossary />} />
                <Route path="learned" element={<Learned />} />
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
