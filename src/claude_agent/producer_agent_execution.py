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
    ThinkingBlock,
    ThinkingConfigEnabled,
    ToolUseBlock,
)
from A_Config import REPO_ROOT, case_name, query_dir, agent_p_output_dir, to_repo_path, pass_num, case_dir, asset_name, asr_transcript_path
PROJECT_ROOT = REPO_ROOT
# cwd must be the repo root (where .claude/ lives) -- Claude Code resolves
# every relative tool path against the project root it finds by walking up
# from cwd, NOT against the literal cwd itself. Setting cwd to Data/ here
# previously caused it to still resolve against the repo root one level up
# (where .claude/agents/producer.md lives), silently dropping "Data" from
# every write path. producer.md no longer names the folder at all -- each
# prompt carries OUTPUT_DIR, built by to_repo_path(agent_p_output_dir()), so
# the path the agent writes to and the path this module later globs are the
# same A_Config getter and cannot drift (including its experiment subfolder).
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

    # Write creates parents on its own, but pre-creating means a run where the
    # agent wrote nowhere fails as an empty glob below, not FileNotFoundError here.
    agent_p_output_dir().mkdir(parents=True, exist_ok=True)

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
            effort="low",
            # summarized (not omitted) so we get a readout of what the model
            # actually weighed re: MAP/R5.1 vs R9.1, instead of a black box
            thinking=ThinkingConfigEnabled(type="enabled", budget_tokens=4096, display="summarized"),
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
            log(f"turn {turn_num} content blocks: {[type(b).__name__ for b in message.content]}")
            for block in message.content:
                if isinstance(block, TextBlock):
                    final_text += block.text
                elif isinstance(block, ThinkingBlock):
                    log(f"thinking:\n{block.thinking}")
                elif isinstance(block, ToolUseBlock):
                    log(f"tool call: {block.name} ({block.input.get('file_path', block.input.get('pattern', ''))})")
                    if block.name == "Write":
                        files_written.append(block.input.get("file_path"))
                        log(f"write content:\n{block.input.get('content', '')}")
                else:
                    log(f"unhandled block type {type(block).__name__}: {block!r}")
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

    out_dir = agent_p_output_dir()
    print(out_dir)
    # Producer writes ONE file — the paper-edit JSON, under producer.md's own
    # fixed name (it has no notion of pass numbers). Everything downstream
    # (flags dict, HTML) is derived from it deterministically, no model involved,
    # and lives in the next cell so it can be re-run on its own (e.g. after
    # editing render_paper_edit.py) without paying for another agent call.
    # producer.md always writes the same base filename regardless of how many
    # times we've called it, so rename here to this pipeline's real naming
    # scheme -- "<asset_name>_<pass>_<draft|revision>.json" (see
    # render_paper_edit.paper_edit_path, the one place that stem is defined)
    # -- using the pass bumped in the notebook each time feedback is given,
    # so passes accumulate on disk instead of colliding.
    from render_paper_edit import paper_edit_path
    output_filename_suffix = "_paper_edit_draft.json" if mode in ("draft", "draft2") else "_paper_edit.json"
    paper_edit_json_path = next(out_dir.glob(f"*{output_filename_suffix}"))

    dated_path = paper_edit_path(mode)
    if dated_path.exists():
        raise FileExistsError(
            f"{dated_path} already exists -- bump A_Config.set_pass(...) before "
            f"calling run_producer_agent again, or you'll overwrite this pass."
        )
    paper_edit_json_path = paper_edit_json_path.rename(dated_path)

    print(f"\npaper_edit_json_path = {paper_edit_json_path}")

    return paper_edit_json_path

def build_producer_prompt_draft1(brief: str, analysis_2d_for_decisions: dict, producer_flags: dict) -> str:
    description_text = (query_dir() / f"{case_name()}_description.txt").read_text(encoding="utf-8")
    objects_text = (query_dir() / f"{case_name()}_objects.txt").read_text(encoding="utf-8")
    transcript_text = asr_transcript_path().read_text(encoding="utf-8")

    places_text  =   (query_dir() / f"{case_name()}_places.txt").read_text(encoding="utf-8")
    people_text  =   (query_dir() / f"{case_name()}_people.json").read_text(encoding="utf-8")
    source_text = f"DESCRIPTION\n{description_text}\nOBJECTS\n{objects_text}\nAUDIO TRANSCRIPT\n{transcript_text}\nPLACES{places_text}\nPEOPLE\n{people_text}"

    return f"""
            MODE: draft

            CASE_NAME: {case_name()}

            OUTPUT_DIR: {to_repo_path(agent_p_output_dir())}

            BRIEF:
            {brief}         

            SOURCE MATERIAL:

            {source_text}

            2D ANALYSIS — deterministic measurement, treat as fact:
            {analysis_2d_for_decisions}

            PRODUCER FLAGS (things you may request by setting True — do not touch anything not listed):
            {producer_flags}

            Produce the paper edit per your instructions.
            """

def build_producer_prompt_draft2(brief: str, analysis_2d_for_decisions: dict, producer_flags: dict) -> str:
    description_text = (query_dir() / f"{case_name()}_description.txt").read_text(encoding="utf-8")
    objects_text = (query_dir() / f"{case_name()}_objects.txt").read_text(encoding="utf-8")
    transcript_text = (query_dir() / f"{case_name()}_transcript_full.txt").read_text(encoding="utf-8")
    places_text  =   (query_dir() / f"{case_name()}_places.txt").read_text(encoding="utf-8")
    people_text  =   (query_dir() / f"{case_name()}_people.json").read_text(encoding="utf-8")
    source_text = f"DESCRIPTION\n{description_text}\nOBJECTS\n{objects_text}\nAUDIO TRANSCRIPT\n{transcript_text}\nPLACES{places_text}\nPEOPLE\n{people_text}"
    # "your draft" is the Producer's own prior draft output -- the one the
    # notebook is now giving feedback on. By the time this is called, the
    # notebook has already bumped pass_num() for the draft2 write about to
    # happen, so that prior draft sits one pass back.
    from render_paper_edit import paper_edit_path
    previous_paper_edit_text = paper_edit_path("draft", pass_n=pass_num() - 1).read_text(encoding="utf-8")
    return f"""
            MODE: draft

            CASE_NAME: {case_name()}

            OUTPUT_DIR: {to_repo_path(agent_p_output_dir())}

            BRIEF:
            {brief}        

            ERROR_CORRECTION: (your draft)
            {previous_paper_edit_text} 
            
            SOURCE MATERIAL:
            {source_text}

            2D ANALYSIS — deterministic measurement, treat as fact:
            {analysis_2d_for_decisions}

            PRODUCER FLAGS (things you may request by setting True — do not touch anything not listed):
            {producer_flags}

            Produce the paper edit per your instructions.
            """

def build_producer_prompt_revision(brief: str, analysis_2d_for_decisions: dict, producer_flags: dict, paper_edit_json_path) -> str:
    description_text = (query_dir() / f"{case_name()}_description.txt").read_text(encoding="utf-8")
    objects_text = (query_dir() / f"{case_name()}_objects.txt").read_text(encoding="utf-8")
    transcript_text = (query_dir() / f"{case_name()}_transcript_full.txt").read_text(encoding="utf-8")
    places_text  =   (query_dir() / f"{case_name()}_places.txt").read_text(encoding="utf-8")
    people_text  =   (query_dir() / f"{case_name()}_people.json").read_text(encoding="utf-8")
    source_text = f"DESCRIPTION\n{description_text}\nOBJECTS\n{objects_text}\nAUDIO TRANSCRIPT\n{transcript_text}\nPLACES{places_text}\nPEOPLE\n{people_text}"
    draft_paper_edit_text = paper_edit_json_path.read_text(encoding="utf-8")

    return f"""
            MODE: revision

            CASE_NAME: {case_name()}

            OUTPUT_DIR: {to_repo_path(agent_p_output_dir())}

            BRIEF:
            {brief}

            SOURCE MATERIAL:
            {source_text}

            2D ANALYSIS deterministic measurement, treat as fact:
            {analysis_2d_for_decisions}

            PRODUCER FLAGS (things you may request by setting True — do not touch anything not listed):
            {producer_flags}

            DRAFT PAPER EDIT (your own informed-pass output, now resolved by the deterministic auto_select step and the Assistant's feasibility verdicts — revise per your instructions):
            {draft_paper_edit_text}

            Produce the paper edit per your instructions.
            """
