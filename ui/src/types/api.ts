// Extension types

export interface PackagePermissionResponse {
  id: string
  display_name: string
  description: string
  risk_level: string
}

export interface ExtensionResponse {
  id: string
  name: string
  version: string
  description: string
  category: string
  capabilities: string[]
  tool_names: string[]
  provides: string[]
  permissions: PackagePermissionResponse[]
  installed: boolean
}

export interface ExtensionListResponse {
  extensions: ExtensionResponse[]
}

export interface InstalledExtensionResponse {
  id: string
  name: string
  version: string
  installed_at: string
  auto_installed: boolean
  capabilities_active: string[]
  permissions_granted: string[]
}

export interface InstalledExtensionListResponse {
  extensions: InstalledExtensionResponse[]
}

// Merged type used by the UI — extends catalog data with auto_installed flag
export interface ExtensionWithStatus extends ExtensionResponse {
  auto_installed: boolean
}

// Planner types

export interface GoalAnalysisResponse {
  required_capabilities: string[]
  risk_level: string
  confidence: number
  reasoning: string
  constraints: string[]
  requires_hitl: boolean
}

export interface RuntimeAgentResponse {
  id: string
  name: string
  description: string
  capabilities: string[]
  tool_names: string[]
  system_prompt: string
  generated: boolean
}

export interface PermissionSummaryResponse {
  id: string
  risk_level: string
}

export interface InstallSuggestionResponse {
  extension_id: string
  name: string
  description: string
  capabilities_provided: string[]
  permissions: PermissionSummaryResponse[]
}

export interface ValidationErrorResponse {
  code: string
  message: string
}

export interface ValidationWarningResponse {
  code: string
  message: string
}

export interface ValidationResultResponse {
  is_valid: boolean
  errors: ValidationErrorResponse[]
  warnings: ValidationWarningResponse[]
}

export interface GuardrailConfigResponse {
  rule_type: string
  config: Record<string, unknown>
  reason: string
}

export interface GeneratePlanResponse {
  plan_id: string
  goal: string
  status: string
  executable: boolean
  task_label: string
  goal_analysis: GoalAnalysisResponse
  selected_pattern: string
  selected_agents: string[]
  runtime_agents: RuntimeAgentResponse[]
  selected_tools: string[]
  guardrails: GuardrailConfigResponse[]
  hitl_required: boolean
  warnings: string[]
  explanation: string
  estimated_complexity: string
  estimated_duration_seconds: number
  validation: ValidationResultResponse
  missing_capabilities: string[]
  install_suggestions: InstallSuggestionResponse[]
  unsupported: boolean
}

export interface ExecutionResponse {
  plan_id: string
  run_id: string
  status: string
  output: string | null
}
