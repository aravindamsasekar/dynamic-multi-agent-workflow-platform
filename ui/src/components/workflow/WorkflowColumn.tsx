import { useEffect, useState } from 'react'
import { usePlannerFlow } from '../../hooks/usePlannerFlow'
import ErrorBanner from '../shared/ErrorBanner'
import ApprovePanel from './ApprovePanel'
import GoalInput from './GoalInput'
import InstallCard from './InstallCard'
import PlanPreview from './PlanPreview'
import ProgressStepper from './ProgressStepper'
import ResultViewer from './ResultViewer'
import SampleGoals from './SampleGoals'

interface WorkflowColumnProps {
  onInstallComplete: () => void
}

export default function WorkflowColumn({ onInstallComplete }: WorkflowColumnProps) {
  const { state, generate, install, approve, reset, clearError } =
    usePlannerFlow(onInstallComplete)

  const { phase, goal, plan, result, error, installEncountered } = state

  // Local textarea value — separate from the committed goal in state
  const [textareaValue, setTextareaValue] = useState('')

  // Sync textarea to empty when flow resets to idle
  useEffect(() => {
    if (phase === 'idle') setTextareaValue('')
  }, [phase])

  // When the flow commits a goal (after generate), reflect it
  useEffect(() => {
    if (goal && textareaValue === '') setTextareaValue(goal)
  }, [goal]) // eslint-disable-line react-hooks/exhaustive-deps

  const handleSubmit = (v: string) => {
    if (v.trim()) generate(v.trim())
  }

  const showPlan =
    plan !== null && phase !== 'idle' && phase !== 'generating'
  const showInstall = phase === 'pending_install' || phase === 'installing'
  const showApprove = phase === 'pending_review'
  const showExecuting = phase === 'executing'
  const showResult = phase === 'done' && result !== null

  return (
    <div>
      {error && <ErrorBanner message={error} onDismiss={clearError} />}

      <GoalInput
        value={textareaValue}
        onChange={setTextareaValue}
        onSubmit={handleSubmit}
        phase={phase}
      />

      <SampleGoals phase={phase} onSelect={setTextareaValue} />

      <ProgressStepper phase={phase} installEncountered={installEncountered} />

      {showPlan && <PlanPreview plan={plan!} />}

      {showInstall && plan && (
        <InstallCard
          suggestions={plan.install_suggestions}
          missingCapabilities={plan.missing_capabilities}
          onInstall={install}
          installing={phase === 'installing'}
        />
      )}

      {(showApprove || showExecuting) && plan && (
        <ApprovePanel
          initialGoal={goal}
          onApprove={approve}
          executing={showExecuting}
        />
      )}

      {showResult && plan && result && (
        <ResultViewer result={result} plan={plan} onReset={reset} />
      )}
    </div>
  )
}
