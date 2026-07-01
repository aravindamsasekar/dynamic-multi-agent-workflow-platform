import Header from './components/layout/Header'
import TwoColumnLayout from './components/layout/TwoColumnLayout'
import MarketplacePanel from './components/marketplace/MarketplacePanel'
import WorkflowColumn from './components/workflow/WorkflowColumn'
import { useExtensions } from './hooks/useExtensions'

export default function App() {
  const extensions = useExtensions()

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      <Header />
      <TwoColumnLayout
        left={<WorkflowColumn onInstallComplete={extensions.refresh} />}
        right={<MarketplacePanel {...extensions} />}
      />
    </div>
  )
}
