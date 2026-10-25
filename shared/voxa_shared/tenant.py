"""Versioned, validated tenant configuration (the single source of truth for an agent's behaviour)."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = 1


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Identity(_M):
    org_name: str = Field(min_length=1, max_length=120)
    agent_name: str = Field(default="Voxa", min_length=1, max_length=60)
    persona: str = "A friendly, efficient customer support specialist."
    tone: str = "warm, concise, professional"
    languages: list[str] = Field(default_factory=lambda: ["en"], min_length=1)
    voice_id: str = "default"
    speaking_rate: float = Field(default=1.0, ge=0.6, le=1.6)


class Scripts(_M):
    greeting: str = "Hi, thanks for calling. How can I help you today?"
    closing: str = "Thanks for calling. Goodbye!"
    clarify: str = "Sorry, I didn't quite catch that. Could you say it again?"
    reprompt: str = "Are you still there?"
    silence_closing: str = "I haven't heard anything, so I'll end the call. Goodbye."
    fallback: list[str] = Field(default_factory=lambda: [
        "I'm not sure about that one. I can connect you with a person if you'd like."])
    recording_prompt: str = "Before we start: may I record this call to improve our service? Please say yes or no."
    recording_yes: str = "Thank you. This call is now being recorded. How can I help?"
    recording_no: str = "No problem, I won't record this call. How can I help?"
    resume_line: str = "Sorry about that, I lost the connection for a moment. Where were we?"
    tool_failure: str = "I'm having trouble reaching that system right now. Would you like me to connect you with a person?"
    fillers: list[str] = Field(default_factory=lambda: [
        "Sure, let me look that up.", "One moment while I check.", "Okay, give me just a second.",
        "Let me pull that up for you."])


class KnowledgeSource(_M):
    type: Literal["url", "pdf", "docx", "csv", "text", "zendesk", "notion", "confluence"]
    uri: str
    name: str | None = None


class ToolParam(_M):
    """JSON-schema fragment for the tool's arguments (OpenAI function-calling format)."""
    type: Literal["object"] = "object"
    properties: dict[str, Any] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)


class OpenAPIRef(_M):
    spec: dict[str, Any] | None = None
    spec_url: str | None = None
    operation_id: str

    @model_validator(mode="after")
    def _one(self) -> "OpenAPIRef":
        if bool(self.spec) == bool(self.spec_url):
            raise ValueError("exactly one of spec / spec_url")
        return self


class ToolDef(_M):
    name: str = Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_]{1,63}$")
    description: str
    kind: Literal["webhook", "openapi", "mcp"] = "webhook"
    parameters: ToolParam = Field(default_factory=ToolParam)
    # webhook
    url: str | None = None                    # may contain {path_params}
    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"] = "POST"
    headers: dict[str, str] = Field(default_factory=dict)  # values may use {{secret:NAME}}
    openapi: OpenAPIRef | None = None
    mcp_server: str | None = None
    side_effect: bool = False                 # True => read back + confirm before executing
    confirm_template: str | None = None       # e.g. "create a ticket titled {title}"
    timeout_s: float = Field(default=3.0, gt=0, le=10)
    expected_latency_ms: int = 200            # > 300 => speak an acknowledgment while it runs
    verify_identity: bool = False             # account lookups: LLM must have verified identity first
    filler: str | None = None                 # spoken while this tool runs (else a rotating generic filler)

    @model_validator(mode="after")
    def _kind_fields(self) -> "ToolDef":
        if self.kind == "webhook" and not self.url:
            raise ValueError("webhook tool needs url")
        if self.kind == "openapi" and not self.openapi:
            raise ValueError("openapi tool needs openapi block")
        if self.kind == "mcp" and not self.mcp_server:
            raise ValueError("mcp tool needs mcp_server")
        return self


class BusinessHours(_M):
    tz: str = "UTC"
    days: list[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4])   # Mon=0
    start: str = "09:00"
    end: str = "17:00"

    @field_validator("start", "end")
    @classmethod
    def _hhmm(cls, v: str) -> str:
        if not re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", v):
            raise ValueError("HH:MM")
        return v


class Escalation(_M):
    human_request_patterns: list[str] = Field(default_factory=lambda: [
        r"\b(speak|talk|connect|transfer)\b.*\b(human|person|agent|representative|someone|manager)\b",
        r"\b(real|live)\s+(person|human|agent)\b", r"\brepresentative\b", r"\boperator\b",
        r"\bhuman\b", r"\b(need|want|get|give me)\b.{0,25}\b(person|agent|manager|supervisor)\b"])
    negative_sentiment_threshold: int = Field(default=3, ge=1)   # cumulative negative score
    max_consecutive_failures: int = Field(default=3, ge=1)
    restricted_topics: list[str] = Field(default_factory=list)
    business_hours: BusinessHours | None = None
    handoff_line: str = "Of course. I'm connecting you with a colleague now. I'll pass along a summary so you don't have to repeat yourself."
    after_hours_line: str = "Our team is offline right now. I can create a ticket and someone will follow up. Would you like that?"
    restricted_line: str = "I'm not able to help with that topic, so let me connect you with someone who can."
    sip_transfer_uri: str | None = None


class TurnParams(_M):
    vad_threshold: float = Field(default=0.5, ge=0.1, le=0.95)
    vad_threshold_speaking: float = Field(default=0.65, ge=0.1, le=0.98)
    vad_min_speech_ms: int = Field(default=96, ge=32, le=500)
    vad_min_silence_ms: int = Field(default=160, ge=32, le=1000)
    endpoint_complete_ms: int = Field(default=300, ge=100, le=1500)
    endpoint_incomplete_ms: int = Field(default=1500, ge=300, le=4000)
    stage1_min_speech_ms: int = Field(default=160, ge=100, le=400)
    interruption_min_words: int = Field(default=2, ge=1, le=6)
    echo_gate_ratio: float = Field(default=0.05, ge=0.0, le=0.3)   # 0 disables the echo gate
    near_field_gate_db: float = Field(default=9.0, ge=0.0, le=30.0)  # ignore background this far below the caller's level; 0 = off
    non_interruptible_phrases: list[str] = Field(default_factory=list)
    reprompt_after_s: float = 8.0
    close_after_s: float = 20.0
    extra_backchannels: list[str] = Field(default_factory=list)


class Guardrails(_M):
    forbidden_topics: list[str] = Field(default_factory=list)
    compliance_disclaimers: list[str] = Field(default_factory=list)   # spoken once, non-interruptible
    pii_policy: Literal["redact", "none"] = "redact"
    max_session_minutes: int = Field(default=30, ge=1, le=240)
    record_calls: bool = False
    retention_days: int = Field(default=30, ge=0, le=3650)


class TenantConfig(_M):
    schema_version: int = SCHEMA_VERSION
    tenant_id: str = Field(pattern=r"^[a-z0-9_]{2,40}$")   # no '-': room names are "<tenant>-<suffix>"
    version: int = Field(default=1, ge=1)
    identity: Identity
    scripts: Scripts = Field(default_factory=Scripts)
    system_prompt_template: str | None = None   # None => platform default (voxa_shared.prompt)
    knowledge: list[KnowledgeSource] = Field(default_factory=list)
    tools: list[ToolDef] = Field(default_factory=list)
    escalation: Escalation = Field(default_factory=Escalation)
    turn: TurnParams = Field(default_factory=TurnParams)
    guardrails: Guardrails = Field(default_factory=Guardrails)
    pronunciations: dict[str, str] = Field(default_factory=dict)
    llm_model: str | None = None

    @field_validator("schema_version")
    @classmethod
    def _sv(cls, v: int) -> int:
        if v > SCHEMA_VERSION:
            raise ValueError(f"config schema {v} is newer than supported {SCHEMA_VERSION}")
        return v

    @model_validator(mode="after")
    def _unique_tools(self) -> "TenantConfig":
        names = [t.name for t in self.tools]
        if len(names) != len(set(names)):
            raise ValueError("duplicate tool names")
        return self

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.model_dump(mode="json"), sort_keys=True).encode()).hexdigest()[:16]


def migrate(raw: dict[str, Any]) -> dict[str, Any]:
    """Upgrade older stored configs to SCHEMA_VERSION (v1 is current: identity)."""
    return raw


def load_config(text: str) -> TenantConfig:
    """Parse YAML or JSON (JSON is valid YAML)."""
    return TenantConfig.model_validate(migrate(yaml.safe_load(text)))
