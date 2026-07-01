import type { Phase } from '../../hooks/usePlannerFlow'

interface SampleGoalsProps {
  onSelect: (goal: string) => void
  phase: Phase
}

const ENABLED_SAMPLES = [
  'Review GitHub pull request octocat/Hello-World #1',
  'Read README.md',
]

const DISABLED_SAMPLES = ['Explain architecture.md']

export default function SampleGoals({ onSelect, phase }: SampleGoalsProps) {
  if (phase !== 'idle' && phase !== 'error') return null

  return (
    <div className="sample-goals">
      <div className="sample-goals-label">Try a sample goal:</div>
      <div className="sample-goals-chips">
        {ENABLED_SAMPLES.map((goal) => (
          <button
            key={goal}
            className="sample-chip"
            onClick={() => onSelect(goal)}
          >
            {goal}
          </button>
        ))}
        {DISABLED_SAMPLES.map((goal) => (
          <button key={goal} className="sample-chip" disabled>
            {goal}
            <span className="coming-soon-badge">Coming Soon</span>
          </button>
        ))}
      </div>
    </div>
  )
}
