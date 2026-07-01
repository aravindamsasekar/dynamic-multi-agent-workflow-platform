import type { ExecutionResponse, GeneratePlanResponse } from '../../types/api'
import CopyButton from '../shared/CopyButton'
import LoadingButton from '../shared/LoadingButton'

interface ResultViewerProps {
  result: ExecutionResponse
  plan: GeneratePlanResponse
  onReset: () => void
}

export default function ResultViewer({ result, plan, onReset }: ResultViewerProps) {
  const output = result.output ?? '(No output returned)'

  return (
    <div className="result-viewer animate-in">
      <div className="result-header">
        <div className="result-header-left">
          <span className="result-icon">✓</span>
          <div>
            <div className="result-title">Execution Completed</div>
            <div className="result-meta">run_id: {result.run_id}</div>
          </div>
        </div>
      </div>

      <div className="result-summary">
        <div className="result-summary-item">
          <div className="result-summary-label">Workflow Pattern</div>
          <div className="result-summary-value">
            {plan.selected_pattern.replace(/_/g, ' ')}
          </div>
        </div>
        <div className="result-summary-item">
          <div className="result-summary-label">Status</div>
          <div className="result-summary-value" style={{ color: 'var(--color-success)' }}>
            {result.status}
          </div>
        </div>
        <div className="result-summary-item">
          <div className="result-summary-label">Plan ID</div>
          <div
            className="result-summary-value"
            style={{ fontFamily: 'var(--font-mono)', fontSize: 11 }}
          >
            {result.plan_id.slice(0, 8)}…
          </div>
        </div>
      </div>

      <div className="result-output-section">
        <div className="result-output-label">Output</div>
        <pre className="result-output-pre">{output}</pre>
      </div>

      <div className="result-actions">
        <CopyButton text={output} />
        <LoadingButton variant="secondary" onClick={onReset}>
          ↺ Start Over
        </LoadingButton>
      </div>
    </div>
  )
}
