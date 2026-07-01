import { useReducer } from 'react'
import { api, errorMessage } from '../api/client'
import type { ExecutionResponse, GeneratePlanResponse } from '../types/api'

export type Phase =
  | 'idle'
  | 'generating'
  | 'pending_install'
  | 'installing'
  | 'pending_review'
  | 'executing'
  | 'done'
  | 'error'

export interface FlowState {
  phase: Phase
  goal: string
  plan: GeneratePlanResponse | null
  result: ExecutionResponse | null
  error: string | null
  installEncountered: boolean
}

type FlowAction =
  | { type: 'GENERATE'; goal: string }
  | { type: 'PLAN_READY'; plan: GeneratePlanResponse }
  | { type: 'INSTALL' }
  | { type: 'INSTALL_DONE'; plan: GeneratePlanResponse }
  | { type: 'APPROVE' }
  | { type: 'EXECUTION_DONE'; result: ExecutionResponse }
  | { type: 'ERROR'; message: string }
  | { type: 'RESET' }
  | { type: 'CLEAR_ERROR' }

const INITIAL_STATE: FlowState = {
  phase: 'idle',
  goal: '',
  plan: null,
  result: null,
  error: null,
  installEncountered: false,
}

function flowReducer(state: FlowState, action: FlowAction): FlowState {
  switch (action.type) {
    case 'GENERATE':
      return {
        ...INITIAL_STATE,
        phase: 'generating',
        goal: action.goal,
      }
    case 'PLAN_READY':
      return {
        ...state,
        plan: action.plan,
        phase: action.plan.status === 'pending_install' ? 'pending_install' : 'pending_review',
        installEncountered:
          state.installEncountered || action.plan.status === 'pending_install',
        error: null,
      }
    case 'INSTALL':
      return { ...state, phase: 'installing' }
    case 'INSTALL_DONE':
      return {
        ...state,
        plan: action.plan,
        phase: action.plan.status === 'pending_install' ? 'pending_install' : 'pending_review',
        installEncountered: true,
        error: null,
      }
    case 'APPROVE':
      return { ...state, phase: 'executing', error: null }
    case 'EXECUTION_DONE':
      return { ...state, phase: 'done', result: action.result }
    case 'ERROR':
      return { ...state, phase: 'error', error: action.message }
    case 'RESET':
      return { ...INITIAL_STATE }
    case 'CLEAR_ERROR':
      return { ...state, error: null, phase: 'idle' }
    default:
      return state
  }
}

export interface UsePlannerFlowResult {
  state: FlowState
  generate: (goal: string) => void
  install: () => void
  approve: (inputData: string) => void
  reset: () => void
  clearError: () => void
}

export function usePlannerFlow(onInstallComplete: () => void): UsePlannerFlowResult {
  const [state, dispatch] = useReducer(flowReducer, INITIAL_STATE)

  const generate = (goal: string) => {
    if (!goal.trim()) return
    dispatch({ type: 'GENERATE', goal })
    api
      .generatePlan(goal)
      .then((plan) => dispatch({ type: 'PLAN_READY', plan }))
      .catch((err: unknown) => dispatch({ type: 'ERROR', message: errorMessage(err) }))
  }

  const install = () => {
    if (!state.plan) return
    const planId = state.plan.plan_id
    dispatch({ type: 'INSTALL' })
    api
      .installAndRegenerate(planId)
      .then((plan) => {
        dispatch({ type: 'INSTALL_DONE', plan })
        onInstallComplete()
      })
      .catch((err: unknown) => dispatch({ type: 'ERROR', message: errorMessage(err) }))
  }

  const approve = (inputData: string) => {
    if (!state.plan) return
    const planId = state.plan.plan_id
    dispatch({ type: 'APPROVE' })
    api
      .approvePlan(planId, inputData)
      .then((result) => dispatch({ type: 'EXECUTION_DONE', result }))
      .catch((err: unknown) => dispatch({ type: 'ERROR', message: errorMessage(err) }))
  }

  const reset = () => dispatch({ type: 'RESET' })
  const clearError = () => dispatch({ type: 'CLEAR_ERROR' })

  return { state, generate, install, approve, reset, clearError }
}
