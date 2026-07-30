import { useId } from 'react';
import { useTranslation } from 'react-i18next';
import { Loader2, ListRestart } from 'lucide-react';

interface Props {
  model: string;
  models: string[];
  canWrite: boolean;
  onChange: (v: string) => void;
  fetchingModels: boolean;
  fetchDisabled: boolean;
  onFetchModels: () => void;
}

export function LlmModelField({ model, models, canWrite, onChange, fetchingModels, fetchDisabled, onFetchModels }: Props) {
  const { t } = useTranslation();
  const modelFieldId = useId();

  return (
    <div className="form-group">
      <label htmlFor={modelFieldId}>{t('llm.model')}</label>
      <div className="llm-row">
        {/* A native datalist filters options by the input's current value, hiding most fetched
            models — render a select once the list is loaded so all of them show. */}
        {models.length > 0 ? (
          <select id={modelFieldId} value={model} disabled={!canWrite} onChange={e => onChange(e.target.value)}>
            {!models.includes(model) && <option value={model}>{model}</option>}
            {models.map(m => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        ) : (
          <input id={modelFieldId} type="text" value={model} disabled={!canWrite} onChange={e => onChange(e.target.value)} />
        )}
        <button className="btn-secondary" onClick={onFetchModels} disabled={fetchingModels || fetchDisabled}>
          {fetchingModels ? <Loader2 size={16} className="animate-spin" /> : <ListRestart size={16} />}
          {t('llm.fetchModels')}
        </button>
      </div>
    </div>
  );
}
