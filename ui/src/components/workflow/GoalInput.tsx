import { useRef } from 'react'
import type { Phase } from '../../hooks/usePlannerFlow'
import LoadingButton from '../shared/LoadingButton'

interface GoalInputProps {
  value: string
  onChange: (v: string) => void
  onSubmit: (goal: string) => void
  phase: Phase
}

const DISABLED_PHASES: Phase[] = [
  'generating',
  'pending_install',
  'installing',
  'pending_review',
  'executing',
  'done',
]

export default function GoalInput({ value, onChange, onSubmit, phase }: GoalInputProps) {
  const ref = useRef<HTMLTextAreaElement>(null)
  const isDisabled = DISABLED_PHASES.includes(phase)
  const isGenerating = phase === 'generating'

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault()
      if (!isDisabled) onSubmit(value)
    }
  }

  return (
    <div className="goal-input-container">
      <label className="goal-label" htmlFor="goal-textarea">
        What would you like to accomplish?
      </label>
      <textarea
        id="goal-textarea"
        ref={ref}
        className="goal-textarea"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={handleKeyDown}
        disabled={isDisabled}
        placeholder="Describe your goal in natural language…"
        rows={3}
      />
      <div className="goal-actions">
        <LoadingButton
          loading={isGenerating}
          loadingText="Analyzing…"
          onClick={() => onSubmit(value)}
          disabled={isDisabled || !value.trim()}
        >
          ▶ Analyze Goal
        </LoadingButton>
        {!isGenerating && !isDisabled && value.trim() === '' && (
          <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
            Enter a goal or choose a sample below
          </span>
        )}
        {!isGenerating && !isDisabled && (
          <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
            Ctrl+Enter to submit
          </span>
        )}
      </div>
    </div>
  )
}
