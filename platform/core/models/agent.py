"""Agent-related models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 4096
    temperature: float = 1.0


class AgentDefinition(BaseModel):
    agent_id: str
    name: str
    description: str = ""
    system_prompt: str
    tool_names: list[str] = []
    llm_config: LLMConfig = Field(default_factory=LLMConfig)


class AgentResult(BaseModel):
    agent_id: str
    output: str
    tool_calls_made: int = 0
