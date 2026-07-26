import asyncio
import json
import sys
import threading
import time
from pathlib import Path
from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
)
from A_Config import case_dir , REPO_ROOT, case_name
out_dir = case_dir() / "012_agent_p_output"
print(out_dir)
PROJECT_ROOT = REPO_ROOT
# cwd must be the repo root (where .claude/ lives) -- Claude Code resolves
# every relative tool path against the project root it finds by walking up
# from cwd, NOT against the literal cwd itself. Setting cwd to Data/ here
# previously caused it to still resolve against the repo root one level up
# (where .claude/agents/producer.md lives), silently dropping "Data" from
# every write path. producer.md's own instructions now say to write to
# "Data/{CASE_NAME}/012_agent_p_output/" (relative to this root), matching
# where Data/<case_name>/012_agent_p_output already lives on disk.
from constitution_view import producer_constitution

def _make_logger():
    t0 = time.perf_counter()
    def log(label):
        print(f"[{time.strftime('%H:%M:%S')} +{time.perf_counter() - t0:6.2f}s] {label}")
    return log


async def _run_producer_agent_async(task_prompt: str, log):
    """Runs the Producer directly — no outer orchestrator/Agent-dispatch layer.
    producer.md is loaded as this session's system prompt, so it IS the
    Producer from the first turn. tools= actually restricts it to
    Read/Grep/Glob/Write (matching its own frontmatter) — allowed_tools alone
    only pre-approves those without restricting the full default set, which
    is how a stray Bash call slipped through in an earlier version."""
    producer_instructions = Path(PROJECT_ROOT, ".claude", "agents", "producer.md").read_text(encoding="utf-8")
    constitution = producer_constitution(Path(PROJECT_ROOT, "src", "claude_agent", "CONSTITUTION.md"))

    #constitution = Path(PROJECT_ROOT, "CONSTITUTION.md").read_text(encoding="utf-8")
    system_prompt = f"{producer_instructions}\n\n# CONSTITUTION (read this before drafting anything)\n{constitution}"

    final_text = ""
    files_written = []

    # --- compute audit: per-turn token usage, so we can see what's actually
    # costing time/money instead of only the final aggregate ---
    turn_num = 0
    totals = {"input_tokens": 0, "output_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}

    log("query: start")
    async for message in query(
        prompt=task_prompt,
        options=ClaudeAgentOptions(
            cwd=PROJECT_ROOT,
            permission_mode="acceptEdits",  # auto-approve file writes, no interactive prompt
            tools=["Read", "Grep", "Glob", "Write"],  # actual restriction, not just allowed_tools
            system_prompt=system_prompt,
            effort="low",  # was unconstrained adaptive thinking — 17190 output tokens on the last run, ~$0.37 of the $0.371 total
        ),
    ):
        if isinstance(message, AssistantMessage):
            turn_num += 1
            u = message.usage or {}
            for k in totals:
                totals[k] += u.get(k, 0) or 0
            log(
                f"turn {turn_num} usage ({message.model}): "
                f"in={u.get('input_tokens', '?')} out={u.get('output_tokens', '?')} "
                f"cache_write={u.get('cache_creation_input_tokens', '?')} "
                f"cache_read={u.get('cache_read_input_tokens', '?')}"
            )
            for block in message.content:
                if isinstance(block, TextBlock):
                    final_text += block.text
                elif isinstance(block, ToolUseBlock):
                    log(f"tool call: {block.name} ({block.input.get('file_path', block.input.get('pattern', ''))})")
                    if block.name == "Write":
                        files_written.append(block.input.get("file_path"))
        elif isinstance(message, ResultMessage):
            log(
                f"result: status={message.subtype} num_turns={message.num_turns} "
                f"duration_ms={message.duration_ms} duration_api_ms={message.duration_api_ms} "
                f"cost_usd={message.total_cost_usd}"
            )
            log(f"result usage (aggregate): {message.usage}")
            log(f"model_usage (per-model breakdown): {message.model_usage}")

    log(
        f"turn totals across {turn_num} turns: in={totals['input_tokens']} "
        f"out={totals['output_tokens']} cache_write={totals['cache_creation_input_tokens']} "
        f"cache_read={totals['cache_read_input_tokens']}"
    )
    log("query: end")
    return final_text, files_written


def run_producer_agent(task_prompt: str, mode: str = "revision"):
    """Sync wrapper. On Windows, Jupyter's kernel loop is a SelectorEventLoop,
    which can't spawn subprocesses — and the SDK needs to spawn the `claude`
    CLI as a subprocess. So this runs the async call on its own thread with a
    fresh ProactorEventLoop, sidestepping the kernel's loop entirely."""
    log = _make_logger()
    log("run_producer_agent: start")
    result = {}
    error = {}

    def _runner():
        log("thread: start")
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        log("event loop: created")
        try:
            result["value"] = loop.run_until_complete(_run_producer_agent_async(task_prompt, log))
        except Exception as e:
            error["value"] = e
        finally:
            loop.close()
        log("thread: end")

    t = threading.Thread(target=_runner)
    t.start()
    t.join()

    log("run_producer_agent: end")

    if "value" in error:
        raise error["value"]

    final_text, files_written = result["value"]

    print("\n=== Files written ===")
    for f in files_written:
        print(" ", f)

    out_dir = case_dir() / "012_agent_p_output"
    print(out_dir)
    # Producer writes ONE file — the paper-edit JSON. Everything downstream
    # (flags dict, HTML) is derived from it deterministically, no model involved,
    # and lives in the next cell so it can be re-run on its own (e.g. after
    # editing render_paper_edit.py) without paying for another agent call.
    output_filename_suffix = "_paper_edit_draft.json" if mode == "draft" else "_paper_edit.json"
    paper_edit_json_path = next(out_dir.glob(f"*{output_filename_suffix}"))
    print(f"\npaper_edit_json_path = {paper_edit_json_path}")
    
    return paper_edit_json_path

def build_producer_prompt_pass1(brief: str, analysis_2d_for_decisions: dict, producer_flags: dict) -> str:
    query_dir = case_dir() / "012_Gemini_outputs"
    description_text = (query_dir / f"{case_name()}_description.txt").read_text(encoding="utf-8")
    objects_text = (query_dir / f"{case_name()}_objects.txt").read_text(encoding="utf-8")
    transcript_text = (query_dir / f"{case_name()}_transcript_full.txt").read_text(encoding="utf-8")
    source_text = f"DESCRIPTION\n{description_text}\nOBJECTS\n{objects_text}\nAUDIO TRANSCRIPT\n{transcript_text}"

    return f"""
            MODE: draft

            CASE_NAME: {case_name()}

            BRIEF:
            {brief}         

            SOURCE MATERIAL:
            {source_text}

            2D ANALYSIS (subject: the cat) — deterministic measurement, treat as fact:
            {analysis_2d_for_decisions}

            AVAILABLE FLAGS (things you may request by setting True — do not touch anything not listed):
            {producer_flags}

            Produce the paper edit per your instructions.
            """

def build_producer_prompt_pass2(brief: str, analysis_2d_for_decisions: dict, producer_flags: dict) -> str:
    query_dir = case_dir() / "012_Gemini_outputs"
    description_text = (query_dir / f"{case_name()}_description.txt").read_text(encoding="utf-8")
    objects_text = (query_dir / f"{case_name()}_objects.txt").read_text(encoding="utf-8")
    transcript_text = (query_dir / f"{case_name()}_transcript_full.txt").read_text(encoding="utf-8")
    source_text = f"DESCRIPTION\n{description_text}\nOBJECTS\n{objects_text}\nAUDIO TRANSCRIPT\n{transcript_text}"
    draft_paper_edit_text = (case_dir() / "012_agent_p_output" / f"{case_name()}_paper_edit_draft.json").read_text(encoding="utf-8")

    return f"""
            MODE: revision

            CASE_NAME: {case_name()}

            BRIEF:
            {brief}

            SOURCE MATERIAL:
            {source_text}

            2D ANALYSIS (subject: the cat) — deterministic measurement, treat as fact:
            {analysis_2d_for_decisions}

            AVAILABLE FLAGS (things you may request by setting True — do not touch anything not listed):
            {producer_flags}

            DRAFT PAPER EDIT (your own informed-pass output, now resolved by the deterministic auto_select step and the Assistant's feasibility verdicts — revise per your instructions):
            {draft_paper_edit_text}

            Produce the paper edit per your instructions.
            """
