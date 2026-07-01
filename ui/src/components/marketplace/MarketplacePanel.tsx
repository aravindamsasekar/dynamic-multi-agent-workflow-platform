import type { UseExtensionsResult } from '../../hooks/useExtensions'
import ExtensionCard from './ExtensionCard'

interface MarketplacePanelProps extends UseExtensionsResult {}

function SkeletonCard() {
  return (
    <div className="skeleton-card">
      <div className="skeleton skeleton-line" style={{ width: '60%' }} />
      <div className="skeleton skeleton-line-sm" />
      <div className="skeleton skeleton-line" style={{ width: '80%', height: 10 }} />
      <div className="skeleton skeleton-line" style={{ width: '45%', height: 10, marginTop: 12 }} />
    </div>
  )
}

export default function MarketplacePanel({
  extensions,
  loading,
  error,
  refresh,
}: MarketplacePanelProps) {
  const installed = extensions.filter((e) => e.installed)
  const available = extensions.filter((e) => !e.installed)

  return (
    <div className="marketplace-panel">
      <div className="marketplace-header">
        <h2>Extension Marketplace</h2>
        <button
          className="marketplace-refresh-btn"
          onClick={refresh}
          disabled={loading}
          title="Refresh"
        >
          {loading ? <span className="spinner spinner-dark" style={{ width: 12, height: 12, borderWidth: 1.5 }} /> : '↺'}
          Refresh
        </button>
      </div>

      <div className="marketplace-body">
        {error && (
          <div className="marketplace-error">
            <div>Could not load extensions: {error}</div>
            <button onClick={refresh}>Retry</button>
          </div>
        )}

        {/* Installed section */}
        <div className="marketplace-section">
          <div className="marketplace-section-title">
            Installed ({loading ? '…' : installed.length})
          </div>
          {loading ? (
            <>
              <SkeletonCard />
              <SkeletonCard />
            </>
          ) : installed.length === 0 ? (
            <p className="marketplace-empty">No extensions installed yet.</p>
          ) : (
            installed.map((ext) => <ExtensionCard key={ext.id} ext={ext} />)
          )}
        </div>

        {/* Available section */}
        <div className="marketplace-section">
          <div className="marketplace-section-title">
            Available ({loading ? '…' : available.length})
          </div>
          {loading ? (
            <SkeletonCard />
          ) : available.length === 0 ? (
            <p className="marketplace-empty">All available extensions are installed.</p>
          ) : (
            available.map((ext) => <ExtensionCard key={ext.id} ext={ext} />)
          )}
          {!loading && available.length > 0 && (
            <p
              style={{
                fontSize: 11,
                color: 'var(--text-muted)',
                marginTop: 8,
                lineHeight: 1.5,
              }}
            >
              Install via the Workflow column by entering a goal that requires these capabilities.
            </p>
          )}
        </div>
      </div>
    </div>
  )
}
