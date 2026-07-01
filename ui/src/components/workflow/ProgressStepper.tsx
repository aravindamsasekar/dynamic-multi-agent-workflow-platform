import type { Phase } from '../../hooks/usePlannerFlow'

interface ProgressStepperProps {
  phase: Phase
  installEncountered: boolean
}

interface Step {
  id: string
  label: string
}

type StepState = 'pending' | 'active' | 'done' | 'error'

function getSteps(installEncountered: boolean): Step[] {
  const steps: Step[] = [
    { id: 'goal', label: 'Goal' },
    { id: 'planning', label: 'Planning' },
  ]
  if (installEncountered) {
    steps.push({ id: 'install', label: 'Extension Install' })
  }
  steps.push(
    { id: 'approval', label: 'Approval' },
    { id: 'execution', label: 'Execution' },
    { id: 'result', label: 'Result' },
  )
  return steps
}

function getActiveStepId(phase: Phase): string {
  switch (phase) {
    case 'idle':
      return 'goal'
    case 'generating':
      return 'planning'
    case 'pending_install':
    case 'installing':
      return 'install'
    case 'pending_review':
      return 'approval'
    case 'executing':
      return 'execution'
    case 'done':
      return 'result'
    case 'error':
      return 'error'
    default:
      return 'goal'
  }
}

function getStepState(
  step: Step,
  steps: Step[],
  activeId: string,
  phase: Phase,
): StepState {
  if (phase === 'error') {
    const activeIdx = steps.findIndex((s) => s.id === activeId)
    const stepIdx = steps.indexOf(step)
    if (stepIdx < activeIdx) return 'done'
    return 'pending'
  }
  if (step.id === activeId) return 'active'
  const activeIdx = steps.findIndex((s) => s.id === activeId)
  const stepIdx = steps.indexOf(step)
  if (stepIdx < activeIdx) return 'done'
  return 'pending'
}

const CHECK = '✓'

export default function ProgressStepper({
  phase,
  installEncountered,
}: ProgressStepperProps) {
  if (phase === 'idle') return null

  const steps = getSteps(installEncountered)
  const activeId = getActiveStepId(phase)

  return (
    <div className="progress-stepper">
      {steps.map((step) => {
        const state = getStepState(step, steps, activeId, phase)
        const circleClass =
          state === 'active'
            ? 'stepper-circle stepper-circle-active'
            : state === 'done'
              ? 'stepper-circle stepper-circle-done'
              : state === 'error'
                ? 'stepper-circle stepper-circle-error'
                : 'stepper-circle stepper-circle-pending'
        const labelClass =
          state === 'active'
            ? 'stepper-label stepper-label-active'
            : state === 'done'
              ? 'stepper-label stepper-label-done'
              : 'stepper-label'
        const stepIdx = steps.indexOf(step) + 1
        return (
          <div
            key={step.id}
            className={`stepper-step ${state}`}
          >
            <div className={circleClass}>
              {state === 'done' ? CHECK : state === 'active' ? '●' : stepIdx}
            </div>
            <span className={labelClass}>{step.label}</span>
          </div>
        )
      })}
    </div>
  )
}
