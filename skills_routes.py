"""
skills_routes.py — FastAPI router for skill storage and execution.

Adds endpoints to the bridge so skills live server-side:
  GET  /skills              — list available skills
  GET  /skills/{name}       — read a skill's markdown
  POST /skills/{name}/run   — execute a skill (builds prompt, calls LLM, parses output)

Skills are markdown files in the SKILLS_DIR (default ./skills).
Each skill has YAML frontmatter + markdown body with mode instructions.
"""

from __future__ import annotations

import os
import re
from typing import Any, Optional, List, Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# Import LLM gateway from the bridge itself
import llm_gateway
from llm_gateway import ChatRequest, ChatMessage, get_router

SKILLS_DIR = os.environ.get("SKILLS_DIR", "./skills")

skills_router = APIRouter(prefix="/skills", tags=["skills"])


# ── request / response models ────────────────────────────────────────────────

class SkillRunRequest(BaseModel):
    mode: str = Field(..., description="plan | advise | debug | review")
    objective: str = Field(..., description="One line: what we're trying to do")
    context: str = Field("", description="Minimal state slice (<300 chars)")
    errors: str = Field("", description="Exact error text / exit codes")
    constraints: str = Field("", description="What's off-limits, environment notes")
    attempted: List[str] = Field(default_factory=list, description="What executor already tried")
    checkpoint_state: Optional[str] = Field(None, description="Current step if re-entering")
    decision_rules: bool = Field(False, description="Ask for reusable rules, not just answers")
    temperature: float = Field(0.3, ge=0.0, le=2.0, description="Lower = stricter format compliance")
    max_tokens: int = Field(800, ge=1, le=4000, description="Target 500-800 for cost")
    provider: str = Field("moonshot", description="LLM provider to use")
    model: Optional[str] = Field("kimi-k3", description="Model name")


class SkillRunResponse(BaseModel):
    skill: str
    mode: str
    content: str
    parsed: Optional[Dict[str, str]] = None
    provider: str
    model: str
    usage: Dict[str, int]
    cost_cents: float
    latency_ms: float
    degraded: bool = False
    missing_sections: Optional[List[str]] = None


class SkillInfo(BaseModel):
    name: str
    description: str
    modes: List[str]


class SkillListResponse(BaseModel):
    skills: List[SkillInfo]


class SkillContentResponse(BaseModel):
    name: str
    frontmatter: Dict[str, Any]
    body: str


# ── helpers ──────────────────────────────────────────────────────────────────

def _list_skill_files() -> List[str]:
    """Return all .md files in SKILLS_DIR."""
    if not os.path.isdir(SKILLS_DIR):
        return []
    return [f for f in os.listdir(SKILLS_DIR) if f.endswith(".md")]


def _parse_frontmatter(text: str) -> tuple[Dict[str, Any], str]:
    """Parse YAML frontmatter from markdown. Returns (frontmatter_dict, body)."""
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            try:
                import yaml
                front = yaml.safe_load(parts[1])
                body = parts[2].strip()
                return (front if front else {}, body)
            except Exception:
                pass
    return {}, text


def _extract_mode_instructions(body: str) -> Dict[str, str]:
    """Extract mode instructions from skill markdown body."""
    instructions = {}
    pattern = r"###\s+`?(\w+)`?\s*\n+(.*?)(?=###\s+\w|\Z)"
    for m in re.finditer(pattern, body, re.DOTALL):
        mode_name = m.group(1).lower()
        section_body = m.group(2).strip()
        instructions[mode_name] = section_body
    return instructions


def _build_prompt(skill_body: str, req: SkillRunRequest) -> str:
    """Build the final prompt for the LLM."""
    mode_instructions = _extract_mode_instructions(skill_body)
    mode_key = req.mode.lower()

    default_instruction = "You are a senior engineer. Respond in structured format. No prose."
    instruction = mode_instructions.get(mode_key, default_instruction)

    lines = [instruction, "", f"Objective: {req.objective}"]
    if req.context:
        lines.append(f"State: {req.context}")
    if req.errors:
        lines.append(f"Errors: {req.errors}")
    if req.constraints:
        lines.append(f"Constraints: {req.constraints}")
    if req.checkpoint_state:
        lines.append(f"Checkpoint: {req.checkpoint_state}")
    if req.attempted:
        lines.append(f"Already tried: {', '.join(req.attempted)}")
    if req.decision_rules:
        lines.append("Include reusable decision rules for similar cases.")
    lines.append("")
    lines.append("Respond only in the specified format. No preamble, no explanation, no prose.")

    prompt = "\n".join(lines)
    if len(prompt) > 600:
        prompt = prompt[:600] + "..."
    return prompt


def _parse_structured(content: str, mode: str) -> Dict[str, Any]:
    """Parse structured output into sections."""
    sections: Dict[str, Any] = {}
    valid_sections = {
        "plan":   ["plan", "risks", "checkpoints", "if_fail", "decision_rules"],
        "advise": ["verdict", "reasoning", "next_steps", "call_back_if"],
        "debug":  ["diagnosis", "root_cause", "fix", "verify", "prevent"],
        "review": ["verdict", "concerns", "strengths", "alternatives", "confidence"],
    }
    valid = set(valid_sections.get(mode, []))

    cleaned = re.sub(r"```[\s\S]*?```", "", content)
    cleaned = cleaned.replace("`", "").replace("*", "")
    cleaned = re.sub(r"#+\s*", "", cleaned)

    current = None
    buffer: List[str] = []

    for line in cleaned.split("\n"):
        m = re.match(r"^([A-Za-z_0-9]+):\s*(.*)", line)
        if m:
            section = m.group(1).lower()
            if section in valid:
                if current:
                    sections[current] = "\n".join(buffer).strip()
                current = section
                buffer = [m.group(2)] if m.group(2) else []
                continue
        if current:
            buffer.append(line)
    if current:
        sections[current] = "\n".join(buffer).strip()

    missing = [s for s in valid_sections.get(mode, []) if s not in sections]
    if missing:
        sections["_degraded"] = True
        sections["_missing"] = missing

    return sections


# ── routes ───────────────────────────────────────────────────────────────────

@skills_router.get("", response_model=SkillListResponse)
async def list_skills():
    """List all available skills with name, description, and supported modes."""
    files = _list_skill_files()
    skills = []
    for fname in files:
        path = os.path.join(SKILLS_DIR, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            front, body = _parse_frontmatter(text)
            name = front.get("name", fname.replace(".md", ""))
            desc = front.get("description", "")
            modes = list(_extract_mode_instructions(body).keys())
            skills.append(SkillInfo(name=name, description=desc, modes=modes))
        except Exception:
            continue
    return {"skills": skills}


@skills_router.get("/{name}", response_model=SkillContentResponse)
async def get_skill(name: str):
    """Read a skill's full markdown content (frontmatter + body)."""
    path = os.path.join(SKILLS_DIR, f"{name}.md")
    if not os.path.exists(path):
        raise HTTPException(404, f'Skill "{name}" not found')
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    front, body = _parse_frontmatter(text)
    return {"name": name, "frontmatter": front, "body": body}


@skills_router.post("/{name}/run", response_model=SkillRunResponse)
async def run_skill(name: str, req: SkillRunRequest):
    """Execute a skill: build prompt from skill instructions + user input, call LLM, parse output."""
    path = os.path.join(SKILLS_DIR, f"{name}.md")
    if not os.path.exists(path):
        raise HTTPException(404, f'Skill "{name}" not found')

    with open(path, "r", encoding="utf-8") as f:
        skill_text = f.read()

    front, body = _parse_frontmatter(skill_text)

    prompt = _build_prompt(body, req)

    router = get_router()
    if not router.available_providers():
        raise HTTPException(503, "No LLM providers configured.")

    chat_req = ChatRequest(
        provider=req.provider,
        model=req.model,
        messages=[ChatMessage(role="user", content=prompt)],
        temperature=req.temperature,
        max_tokens=req.max_tokens,
    )

    result = await router.chat(chat_req)

    parsed = _parse_structured(result.content, req.mode)
    degraded = parsed.pop("_degraded", False)
    missing = parsed.pop("_missing", None)

    return SkillRunResponse(
        skill=name,
        mode=req.mode,
        content=result.content,
        parsed=parsed if not degraded else None,
        provider=result.provider,
        model=result.model,
        usage=result.usage,
        cost_cents=result.cost_cents,
        latency_ms=result.latency_ms,
        degraded=degraded,
        missing_sections=missing,
    )
