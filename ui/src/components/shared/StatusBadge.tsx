interface StatusBadgeProps {
  value: string
  label?: string
  className?: string
}

const STATUS_CLASS: Record<string, string> = {
  pending_review: 'badge-pending-review',
  pending_install: 'badge-pending-install',
  executed: 'badge-executed',
  completed: 'badge-completed',
  rejected: 'badge-rejected',
  installed: 'badge-installed',
  auto_installed: 'badge-auto',
  available: 'badge-available',
  generated: 'badge-generated',
  low: 'badge-low',
  medium: 'badge-medium',
  high: 'badge-high',
  critical: 'badge-critical',
}

const CATEGORY_CLASS: Record<string, string> = {
  github: 'badge-cat-github',
  knowledge: 'badge-cat-knowledge',
  filesystem: 'badge-cat-filesystem',
}

const STATUS_LABELS: Record<string, string> = {
  pending_review: 'Pending Review',
  pending_install: 'Install Required',
  executed: 'Completed',
  completed: 'Completed',
  rejected: 'Rejected',
  installed: 'Installed',
  auto_installed: '✦ Auto-installed',
  available: 'Available',
  generated: 'Generated Runtime Agent',
}

export default function StatusBadge({ value, label, className = '' }: StatusBadgeProps) {
  const badgeClass =
    STATUS_CLASS[value] ?? CATEGORY_CLASS[value] ?? 'badge-available'
  const displayLabel = label ?? STATUS_LABELS[value] ?? value
  return (
    <span className={`badge ${badgeClass} ${className}`}>
      {displayLabel}
    </span>
  )
}
