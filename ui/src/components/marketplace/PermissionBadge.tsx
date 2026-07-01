interface PermissionBadgeProps {
  id: string
  riskLevel: string
}

const RISK_CLASS: Record<string, string> = {
  low: 'badge-low',
  medium: 'badge-medium',
  high: 'badge-high',
  critical: 'badge-critical',
}

export default function PermissionBadge({ id, riskLevel }: PermissionBadgeProps) {
  const riskClass = RISK_CLASS[riskLevel] ?? 'badge-available'
  return (
    <div className="permission-row">
      <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>{id}</span>
      <span className={`badge ${riskClass}`} style={{ fontSize: 10, padding: '2px 6px' }}>
        {riskLevel}
      </span>
    </div>
  )
}
