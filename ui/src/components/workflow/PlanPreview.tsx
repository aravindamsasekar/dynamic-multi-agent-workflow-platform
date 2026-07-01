import type { GeneratePlanResponse } from '../../types/api'
import StatusBadge from '../shared/StatusBadge'

interface PlanPreviewProps {
  plan: GeneratePlanResponse
}

const MAX_CHIPS = 6

function ChipList({
  items,
  max,
  chipClass,
}: {
  items: string[]
  max?: number
  chipClass: string
}) {
  const limit = max ?? MAX_CHIPS
  const visible = items.slice(0, limit)
  const more = items.length - limit
  return (
    <div className="chip-list">
      {visible.map((item) => (
        <span key={item} className={`chip ${chipClass}`}>
          {item}
        </span>
      ))}
      {more > 0 && <span className="chip-more">+{more} more</span>}
    </div>
  )
}

function formatPattern(p: string) {
  return p.replace(/_/g, ' ')
}

function formatConfidence(c: number) {
  return `${Math.round(c * 100)}%`
}

export default function PlanPreview({ plan }: PlanPreviewProps) {
  const isPendingInstall = plan.status === 'pending_install'
  const allAgents = [
    ...plan.selected_agents,
    ...plan.runtime_agents.map((r) => r.id),
  ]
  const agentsGreyed = isPendingInstall && plan.selected_agents.length === 0

  return (
    <div className="plan-preview animate-in">
      <div className="plan-preview-header">
        <span className="plan-preview-title">Workflow Plan</span>
        <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
          <StatusBadge value={plan.status} />
        </div>
      </div>

      <div className="plan-preview-body">
        {/* Workflow summary */}
        <div className="plan-section">
          <div className="plan-section-title">Workflow</div>
          <div className="plan-workflow-grid">
            <div className="plan-stat">
              <div className="plan-stat-label">Pattern</div>
              <div className="plan-stat-value plan-stat-value-mono">
                {formatPattern(plan.selected_pattern)}
              </div>
            </div>
            <div className="plan-stat">
              <div className="plan-stat-label">Complexity</div>
              <div className="plan-stat-value">{plan.estimated_complexity}</div>
            </div>
            <div className="plan-stat">
              <div className="plan-stat-label">Confidence</div>
              <div className="plan-stat-value">
                {formatConfidence(plan.goal_analysis.confidence)}
              </div>
            </div>
            <div className="plan-stat">
              <div className="plan-stat-label">Risk</div>
              <div className="plan-stat-value">
                <StatusBadge value={plan.goal_analysis.risk_level} />
              </div>
            </div>
          </div>
        </div>

        {/* Capabilities */}
        {plan.goal_analysis.required_capabilities.length > 0 && (
          <div className="plan-section">
            <div className="plan-section-title">Required Capabilities</div>
            <ChipList
              items={plan.goal_analysis.required_capabilities}
              chipClass="chip-capability"
            />
          </div>
        )}

        {/* Agents */}
        {(allAgents.length > 0 || plan.runtime_agents.length > 0) && (
          <div className="plan-section" style={{ opacity: agentsGreyed ? 0.45 : 1 }}>
            <div className="plan-section-title">Agents</div>
            {agentsGreyed && (
              <p style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 6 }}>
                Pending installation
              </p>
            )}
            <div className="chip-list">
              {plan.selected_agents.map((a) => (
                <span key={a} className="chip chip-agent">
                  {a}
                </span>
              ))}
              {plan.runtime_agents.map((r) => (
                <span
                  key={r.id}
                  className="chip chip-agent"
                  style={{ display: 'inline-flex', alignItems: 'center', gap: 5 }}
                >
                  {r.name}
                  {r.generated && (
                    <span className="badge badge-generated" style={{ fontSize: 9, padding: '1px 5px' }}>
                      Generated Runtime Agent
                    </span>
                  )}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Tools */}
        {plan.selected_tools.length > 0 && (
          <div className="plan-section" style={{ opacity: agentsGreyed ? 0.45 : 1 }}>
            <div className="plan-section-title">Tools</div>
            <ChipList items={plan.selected_tools} chipClass="chip-tool" />
          </div>
        )}

        {/* Decision */}
        <div className="plan-section">
          <div className="plan-section-title">Decision</div>
          <p className="plan-explanation">{plan.explanation}</p>
          {plan.goal_analysis.reasoning && (
            <div className="plan-reasoning">{plan.goal_analysis.reasoning}</div>
          )}
          {plan.warnings.length > 0 && (
            <div className="plan-warnings">
              {plan.warnings.map((w, i) => (
                <div key={i} className="plan-warning-item">
                  <span>⚠</span>
                  <span>{w}</span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Validation errors */}
        {plan.validation.errors.length > 0 && (
          <div className="plan-section">
            <div className="plan-section-title" style={{ color: 'var(--color-error)' }}>
              Validation Errors
            </div>
            <div className="plan-validation-errors">
              {plan.validation.errors.map((e) => (
                <div key={e.code} className="plan-validation-error">
                  <strong>[{e.code}]</strong> {e.message}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
