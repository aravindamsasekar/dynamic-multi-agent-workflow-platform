import { useState } from 'react'
import LoadingButton from '../shared/LoadingButton'

interface ApprovePanelProps {
  initialGoal: string
  onApprove: (inputData: string) => void
  executing: boolean
}

export default function ApprovePanel({
  initialGoal,
  onApprove,
  executing,
}: ApprovePanelProps) {
  const [inputData, setInputData] = useState(initialGoal)

  return (
    <div className="approve-panel animate-in">
      <div className="approve-panel-title">Approve & Run</div>
      <label className="approve-input-label" htmlFor="approve-input">
        Execution Input
      </label>
      <textarea
        id="approve-input"
        className="approve-textarea"
        value={inputData}
        onChange={(e) => setInputData(e.target.value)}
        disabled={executing}
        rows={3}
      />
      <p className="approve-note">
        This input is passed as context to the agents. Edit if needed before approving.
      </p>
      <LoadingButton
        loading={executing}
        loadingText="Running…"
        onClick={() => onApprove(inputData)}
        disabled={executing}
        style={{ width: '100%' }}
      >
        ▶ Approve &amp; Run
      </LoadingButton>
    </div>
  )
}
