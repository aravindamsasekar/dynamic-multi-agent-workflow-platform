import { useState } from 'react'
import type { ExtensionWithStatus } from '../../types/api'
import PermissionBadge from './PermissionBadge'

interface ExtensionCardProps {
  ext: ExtensionWithStatus
}

const MAX_CAPS = 4
const MAX_TOOLS = 4

function ChipList({
  items,
  max,
  chipClass,
}: {
  items: string[]
  max: number
  chipClass: string
}) {
  const visible = items.slice(0, max)
  const more = items.length - max
  return (
    <div className="chip-list" style={{ marginTop: 4 }}>
      {visible.map((item) => (
        <span key={item} className={`chip ${chipClass}`}>
          {item}
        </span>
      ))}
      {more > 0 && <span className="chip-more">+{more} more</span>}
    </div>
  )
}

const CAT_BADGE: Record<string, string> = {
  github: 'badge-cat-github',
  knowledge: 'badge-cat-knowledge',
  filesystem: 'badge-cat-filesystem',
}

export default function ExtensionCard({ ext }: ExtensionCardProps) {
  const [toolsOpen, setToolsOpen] = useState(false)
  const catClass = CAT_BADGE[ext.category] ?? 'badge-cat-default'

  return (
    <div className="extension-card">
      <div className="extension-card-header">
        <div>
          <div className="extension-card-title">{ext.name}</div>
          <div className="extension-card-version">v{ext.version}</div>
        </div>
        <span className={`badge ${catClass}`}>{ext.category}</span>
      </div>

      <div className="extension-card-badges">
        {ext.installed && !ext.auto_installed && (
          <span className="badge badge-installed">● Installed</span>
        )}
        {ext.auto_installed && (
          <span className="badge badge-auto">✦ Auto-installed</span>
        )}
        {!ext.installed && (
          <span className="badge badge-available">Available</span>
        )}
      </div>

      <p className="extension-card-desc">{ext.description}</p>

      {ext.capabilities.length > 0 && (
        <>
          <div className="extension-card-section-label">Capabilities</div>
          <ChipList
            items={ext.capabilities}
            max={MAX_CAPS}
            chipClass="chip-capability"
          />
        </>
      )}

      {ext.tool_names.length > 0 && (
        <>
          <button
            className="extension-tools-toggle"
            onClick={() => setToolsOpen((o) => !o)}
          >
            {toolsOpen ? '▾' : '▸'} Tools ({ext.tool_names.length})
          </button>
          {toolsOpen && (
            <ChipList
              items={ext.tool_names}
              max={MAX_TOOLS}
              chipClass="chip-tool"
            />
          )}
        </>
      )}

      {ext.permissions.length > 0 && (
        <div className="extension-card-permissions">
          <div className="extension-card-section-label">Permissions</div>
          {ext.permissions.map((p) => (
            <PermissionBadge key={p.id} id={p.id} riskLevel={p.risk_level} />
          ))}
        </div>
      )}
    </div>
  )
}
