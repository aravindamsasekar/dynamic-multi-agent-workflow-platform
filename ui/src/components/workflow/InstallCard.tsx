import type { InstallSuggestionResponse } from '../../types/api'
import LoadingButton from '../shared/LoadingButton'

interface InstallCardProps {
  suggestions: InstallSuggestionResponse[]
  missingCapabilities: string[]
  onInstall: () => void
  installing: boolean
}

const RISK_CLASS: Record<string, string> = {
  low: 'badge-low',
  medium: 'badge-medium',
  high: 'badge-high',
  critical: 'badge-critical',
}

export default function InstallCard({
  suggestions,
  missingCapabilities,
  onInstall,
  installing,
}: InstallCardProps) {
  if (suggestions.length === 0) return null

  return (
    <div className="install-card animate-in">
      <div className="install-card-header">
        <span className="install-card-icon">⚠</span>
        <div>
          <div className="install-card-title">Installation Required</div>
          <div className="install-card-subtitle">
            Install the extension below to continue.
          </div>
        </div>
      </div>

      <div className="install-card-body">
        {missingCapabilities.length > 0 && (
          <div className="install-card-missing">
            <span style={{ color: 'var(--text-muted)' }}>Missing capabilities:</span>
            <div className="chip-list" style={{ display: 'inline-flex', marginTop: 0 }}>
              {missingCapabilities.map((c) => (
                <span key={c} className="chip chip-capability">
                  {c}
                </span>
              ))}
            </div>
          </div>
        )}

        {suggestions.map((s) => (
          <div key={s.extension_id} className="install-extension-box">
            <div className="install-ext-name">{s.name}</div>
            <p className="install-ext-desc">{s.description}</p>

            {s.capabilities_provided.length > 0 && (
              <>
                <div className="install-ext-section-label">Capabilities provided</div>
                <div className="chip-list">
                  {s.capabilities_provided.map((c) => (
                    <span key={c} className="chip chip-capability">
                      {c}
                    </span>
                  ))}
                </div>
              </>
            )}

            {s.permissions.length > 0 && (
              <>
                <div className="install-ext-section-label">Permissions required</div>
                {s.permissions.map((p) => (
                  <div
                    key={p.id}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      fontSize: 12,
                      color: 'var(--text-secondary)',
                      padding: '3px 0',
                    }}
                  >
                    <span>{p.id}</span>
                    <span
                      className={`badge ${RISK_CLASS[p.risk_level] ?? 'badge-available'}`}
                      style={{ fontSize: 10, padding: '2px 6px' }}
                    >
                      {p.risk_level}
                    </span>
                  </div>
                ))}
              </>
            )}
          </div>
        ))}
      </div>

      <div className="install-card-footer">
        <LoadingButton
          variant="install"
          loading={installing}
          loadingText="Installing…"
          onClick={onInstall}
          style={{ width: '100%' }}
        >
          Install Extension &amp; Continue
        </LoadingButton>
      </div>
    </div>
  )
}
