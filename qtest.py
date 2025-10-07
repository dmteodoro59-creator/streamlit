# app.py
# Streamlit/CLI AI Agent + Chat (OpenRouter via Tongyi DeepResearch 30B A3B)
# -------------------------------------------------------------------------------------------------
# Why this rewrite?
# You reported: SystemExit: 2 raised from run_cli when no API key was provided.
# Fixes:
# 1) **No hard exits** in CLI mode: Missing API key no longer calls `sys.exit(2)`; we print a helpful
#    message and **return gracefully**. This prevents the SystemExit error you're seeing.
# 2) API key is now **checked lazily** only when a network call is actually needed (message/goal given).
# 3) Added more **unit tests** to ensure run_cli doesn’t raise SystemExit on common no-API scenarios,
#    and to increase coverage of parsers and file extraction fallbacks.
#
# Dual-mode app:
# - Streamlit UI when available (run with `streamlit run app.py`).
# - CLI fallback when Streamlit is missing or `--cli` is used.
#
# Streamlit features:
# - Tabs: Chat, Agent Runner
# - File uploads (txt, md, pdf, docx, csv) used as context
# - Uses OpenRouter model: alibaba/tongyi-deepresearch-30b-a3b:free
# - History for chat/agent runs + token usage footer
#
# CLI features:
# - Chat:   python app.py --cli --message "..." [--file ...]
# - Agent:  python app.py --cli --goal "..." --max-steps 5 [--file ...]
# - API key via --api-key or OPENROUTER_API_KEY env var (checked only when needed)
#
# Tests:
# - Run: python app.py --run-tests  (no network required)
# -------------------------------------------------------------------------------------------------

import argparse
import io
import json
import os
import sys
import time
import unittest
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# --- Optional deps ---
try:
    import requests  # required for API calls
except Exception:  # pragma: no cover
    requests = None

try:
    import pandas as pd
except Exception:
    pd = None

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    import docx  # python-docx
except Exception:
    docx = None

# Try Streamlit import, but allow fallback to CLI
try:
    import streamlit as st  # type: ignore
    _HAS_STREAMLIT = True
except Exception:
    st = None  # type: ignore
    _HAS_STREAMLIT = False

# -------------------------------------------------------------------------------------------------
# Constants & Datatypes
# -------------------------------------------------------------------------------------------------
DEFAULT_MODEL = "alibaba/tongyi-deepresearch-30b-a3b:free"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

@dataclass
class LLMResponse:
    content: str
    usage: Dict[str, Any] = field(default_factory=dict)

# -------------------------------------------------------------------------------------------------
# Utilities independent of Streamlit (so they work in CLI/tests)
# -------------------------------------------------------------------------------------------------

def detect_streamlit_run() -> bool:
    """Return True if running inside Streamlit's runtime."""
    if not _HAS_STREAMLIT:
        return False
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx  # type: ignore
        return get_script_run_ctx() is not None
    except Exception:
        return False


def extract_text_from_bytes(data: bytes, filename: str, mime: str = "") -> Dict[str, Any]:
    """Return a dict: {name, type, text, df?}. Supports txt/md/csv/pdf/docx. Falls back to best-effort decode."""
    name = filename
    lower = name.lower()

    # Text/Markdown
    if mime in ("text/plain", "text/markdown") or lower.endswith((".txt", ".md")):
        try:
            return {"name": name, "type": "text", "text": data.decode(errors="ignore")}
        except Exception as e:
            return {"name": name, "type": "text", "text": f"[text decode error] {e}"}

    # CSV
    if lower.endswith(".csv"):
        if pd is None:
            try:
                return {"name": name, "type": "csv", "text": data.decode(errors="ignore")}
            except Exception as e:
                return {"name": name, "type": "csv", "text": f"[csv decode error] {e}"}
        try:
            df = pd.read_csv(io.BytesIO(data))  # type: ignore
            text = df.to_csv(index=False)
            return {"name": name, "type": "csv", "text": text, "df": df}
        except Exception as e:
            return {"name": name, "type": "csv", "text": f"[CSV read error] {e}"}

    # PDF
    if lower.endswith(".pdf") and PdfReader is not None:
        try:
            reader = PdfReader(io.BytesIO(data))
            pages = []
            for p in reader.pages:
                try:
                    pages.append(p.extract_text() or "")
                except Exception:
                    pages.append("")
            return {"name": name, "type": "pdf", "text": "\n\n".join(pages)}
        except Exception as e:
            return {"name": name, "type": "pdf", "text": f"[PDF read error] {e}"}

    # DOCX
    if lower.endswith(".docx") and docx is not None:
        try:
            d = docx.Document(io.BytesIO(data))
            paras = [p.text for p in d.paragraphs]
            return {"name": name, "type": "docx", "text": "\n".join(paras)}
        except Exception as e:
            return {"name": name, "type": "docx", "text": f"[DOCX read error] {e}"}

    # Fallback: best-effort text
    try:
        return {"name": name, "type": "binary", "text": data.decode(errors="ignore")}
    except Exception:
        return {"name": name, "type": "binary", "text": "[Unsupported file type or binary content]"}


def safe_parse_steps(raw: str, max_steps: int) -> List[str]:
    """Parse planning output into a list of steps (JSON object/list or bulleted lines)."""
    steps: List[str] = []
    s = (raw or "").strip()
    if not s:
        return steps
    try:
        data = json.loads(s)
        if isinstance(data, dict) and "steps" in data and isinstance(data["steps"], list):
            steps = [str(x) for x in data["steps"]]
        elif isinstance(data, list):
            steps = [str(x) for x in data]
    except Exception:
        for line in s.splitlines():
            t = line.strip("- •* \t").strip()
            if t:
                steps.append(t)
    steps = [x.strip() for x in steps if x and x.strip()]
    return steps[: max_steps or len(steps)]


def call_openrouter(messages: List[Dict[str, str]], api_key: str, model: str = DEFAULT_MODEL,
                    system_prompt: Optional[str] = None, temperature: float = 0.2,
                    response_format: Optional[Dict[str, Any]] = None,
                    timeout: int = 120) -> LLMResponse:
    if requests is None:
        raise RuntimeError("`requests` not installed. Install it with `pip install requests`.")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "model": model,
        "messages": ([] if not system_prompt else [{"role": "system", "content": system_prompt}]) + messages,
        "temperature": temperature,
    }
    if response_format:
        payload["response_format"] = response_format
    r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    usage = data.get("usage", {})
    return LLMResponse(content=content, usage=usage)

# -------------------------------------------------------------------------------------------------
# CLI Implementation (no Streamlit required)
# -------------------------------------------------------------------------------------------------

def _read_context_files(paths: Optional[List[str]]) -> List[Dict[str, Any]]:
    files: List[Dict[str, Any]] = []
    for path in (paths or []):
        try:
            with open(path, "rb") as f:
                data = f.read()
            files.append(extract_text_from_bytes(data, os.path.basename(path)))
        except Exception as e:
            files.append({"name": os.path.basename(path), "type": "error", "text": f"[read error] {e}"})
    return files


def _build_context(files: List[Dict[str, Any]], limit_chars: int = 4000) -> str:
    parts = []
    for u in files:
        parts.append(f"[FILE: {u['name']}]\n{(u.get('text') or '')[:limit_chars]}")
    return "\n\n".join(parts)


def _need_api_key(args: argparse.Namespace) -> bool:
    return bool(args.message or args.goal)


def _get_api_key(args: argparse.Namespace) -> Optional[str]:
    key = args.api_key or os.environ.get("OPENROUTER_API_KEY") or ""
    if not key:
        print("ERROR: Provide an API key via --api-key or OPENROUTER_API_KEY env var.")
        return None
    return key


def run_cli(args: argparse.Namespace) -> None:
    # If there's nothing to do, guide the user and return gracefully.
    if not _need_api_key(args):
        print(
            "CLI help:\n"
            "  Chat : python app.py --cli --message \"Hello\" --file notes.txt\n"
            "  Agent: python app.py --cli --goal \"Summarize report.pdf\" --file report.pdf --max-steps 5\n"
            "Set your API key with --api-key or OPENROUTER_API_KEY."
        )
        return

    api_key = _get_api_key(args)
    if not api_key:
        # Missing API key – do not raise SystemExit; just return.
        return

    files = _read_context_files(args.file)

    if args.message:
        system_prompt = (
            "You are a helpful AI assistant. If FILE context is provided, use it to answer concisely. "
            "Cite filenames inline like [filename] where relevant."
        )
        user_msg = args.message
        ctx = _build_context(files, 4000)
        if ctx:
            user_msg = f"Context from uploads (may be truncated):\n{ctx}\n\nUser question: {args.message}"
        print("\n# Chat Response\n")
        try:
            resp = call_openrouter(
                messages=[{"role": "user", "content": user_msg}],
                api_key=api_key,
                model=args.model,
                system_prompt=system_prompt,
                temperature=args.temperature,
            )
            print(resp.content.strip())
            if resp.usage:
                print("\n[usage]", resp.usage)
        except Exception as e:
            print(f"API error: {e}")

    if args.goal:
        plan_sys = (
            "You are a pragmatic project planner. Given a goal and optional context, "
            "return a concise JSON array of atomic steps. Each step should be actionable and < 20 words."
        )
        ctx = _build_context(files, 2500)
        plan_user = (
            f"Goal: {args.goal}\n\nContext (may be truncated):\n{ctx}\n\nReturn JSON array only, no prose. Limit to {args.max_steps} steps."
        )
        print("\n# Agent Plan\n")
        try:
            plan_resp = call_openrouter(
                messages=[{"role": "user", "content": plan_user}],
                api_key=api_key,
                model=args.model,
                system_prompt=plan_sys,
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            steps = safe_parse_steps(plan_resp.content, args.max_steps)
        except Exception as e:
            print(f"Planning API error: {e}")
            steps = []

        if not steps:
            print("No steps planned. Try adjusting your goal.")
            return

        for i, s in enumerate(steps, 1):
            print(f"{i}. {s}")

        artifacts: List[Dict[str, Any]] = []
        print("\n# Agent Execution\n")
        for i, step in enumerate(steps, 1):
            parts = []
            if files:
                parts.append(_build_context(files, 3500))
            if artifacts:
                for j, a in enumerate(artifacts, 1):
                    preview = a.get("content", "")
                    parts.append(f"[ARTIFACT {j}: {a.get('title','step output')}]\n{preview[:3000]}")
            ctx2 = "\n\n".join(parts)

            exec_sys = (
                "You are an expert operator. Perform the requested step using the provided context. "
                "Return a high-quality, self-contained result."
            )
            user = f"Step to perform: {step}\n\nRelevant context (truncated):\n{ctx2}"
            try:
                exec_resp = call_openrouter(
                    messages=[{"role": "user", "content": user}],
                    api_key=api_key,
                    model=args.model,
                    system_prompt=exec_sys,
                    temperature=args.temperature,
                )
                content = exec_resp.content.strip()
            except Exception as e:
                content = f"[execution API error] {e}"
            artifacts.append({"title": step[:60], "content": content})
            print(f"\n## Step {i} Output\n{content}\n")

        print("\n# Suggested Next Steps\n")
        summ = "\n\n".join([f"[{a['title']}]\n{a['content'][:1200]}" for a in artifacts])
        adv_sys = "You are a strategic advisor. Suggest concrete, high-leverage next moves (bulleted)."
        adv_user = f"Objective: {args.goal}\n\nWhat we accomplished (truncated):\n{summ}\n\nSuggest 3-5 concise, high-impact next steps."
        try:
            ideas_resp = call_openrouter(
                messages=[{"role": "user", "content": adv_user}],
                api_key=api_key,
                model=args.model,
                system_prompt=adv_sys,
                temperature=0.3,
            )
            lines = [ln.strip("- •* \t") for ln in ideas_resp.content.splitlines() if ln.strip()]
            ideas = [ln for ln in lines if len(ln) > 3][:5]
        except Exception as e:
            ideas = [f"[advice API error] {e}"]
        for it in ideas:
            print(f"- {it}")

# -------------------------------------------------------------------------------------------------
# Streamlit Implementation
# -------------------------------------------------------------------------------------------------

def run_streamlit_app() -> None:  # Only called when running under Streamlit
    st.set_page_config(page_title="AI Agent + Chat (Tongyi via OpenRouter)", page_icon="🤖", layout="wide")

    # -------- Session state --------
    if "chat_history" not in st.session_state:
        st.session_state.chat_history: List[Dict[str, str]] = []
    if "uploads" not in st.session_state:
        st.session_state.uploads: List[Dict[str, Any]] = []
    if "usage" not in st.session_state:
        st.session_state.usage = {"total_prompt_tokens": 0, "total_completion_tokens": 0, "total_requests": 0}
    if "agent_runs" not in st.session_state:
        st.session_state.agent_runs: List[Dict[str, Any]] = []

    def _add_usage(usage: Optional[Dict[str, Any]]):
        if not usage:
            return
        st.session_state.usage["total_prompt_tokens"] += usage.get("prompt_tokens", 0)
        st.session_state.usage["total_completion_tokens"] += usage.get("completion_tokens", 0)
        st.session_state.usage["total_requests"] += 1

    # -------- Sidebar --------
    st.sidebar.title("🔑 Configuration")
    api_key = st.sidebar.text_input("OpenRouter API Key", value=os.environ.get("OPENROUTER_API_KEY", ""), type="password")
    model = st.sidebar.text_input("Model", value=DEFAULT_MODEL)

    tab_help = st.sidebar.expander("ℹ️ Tips")
    with tab_help:
        st.markdown(
            """
            - Your uploads become context for both Chat and the Agent.
            - The Agent first *plans* a task list, then *executes* steps by generating outputs.
            - After completion, it proposes **3–5 next-step ideas**.
            - Token usage is tallied at the bottom of the page.
            """
        )

    st.sidebar.markdown("---")
    uploads = st.sidebar.file_uploader(
        "Upload context files", type=["txt", "md", "pdf", "docx", "csv"], accept_multiple_files=True
    )
    if uploads:
        added = []
        for up in uploads:
            try:
                info = extract_text_from_bytes(up.read(), up.name, getattr(up, "type", ""))
                st.session_state.uploads.append(info)
                added.append(info["name"])
            except Exception as e:
                st.sidebar.error(f"{up.name}: {e}")
        if added:
            st.sidebar.success(f"Added: {', '.join(added)}")

    if st.session_state.uploads:
        with st.sidebar.expander("📎 Current uploads"):
            for u in st.session_state.uploads:
                st.caption(f"• {u['name']} ({u['type']}) – {len(u.get('text',''))} chars")

    # -------- Tabs --------
    chat_tab, agent_tab = st.tabs(["💬 Chatbot", "🧭 Agent Runner"])

    # -------- Chatbot Tab --------
    with chat_tab:
        st.subheader("AI Chatbot")

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])  

        prompt = st.chat_input("Type your message…")
        if prompt:
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            context_blurbs = []
            for u in st.session_state.uploads:
                context_blurbs.append(f"[FILE: {u['name']}]\n{(u.get('text') or '')[:4000]}")
            ctx = ("\n\n".join(context_blurbs)) if context_blurbs else ""

            sys_prompt = (
                "You are a helpful AI assistant. If FILE context is provided, use it to answer concisely. "
                "Cite filenames inline like [filename] where relevant."
            )
            user_msg = prompt if not ctx else f"Context from uploads (may be truncated):\n{ctx}\n\nUser question: {prompt}"

            try:
                llm = call_openrouter(
                    messages=[{"role": "user", "content": user_msg}],
                    api_key=api_key,
                    model=model,
                    system_prompt=sys_prompt,
                    temperature=0.2,
                )
                _add_usage(llm.usage)
                reply = llm.content.strip()
            except Exception as e:
                reply = f"⚠️ API error: {e}"

            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.markdown(reply)

    # -------- Agent Tab --------
    with agent_tab:
        st.subheader("Agent: Plan → Execute → Propose Next Steps")

        with st.form("agent_form"):
            goal = st.text_area(
                "What objective should the agent accomplish?",
                placeholder="e.g., Summarize the uploaded PDF and draft a 1-page brief with key insights.",
            )
            max_steps = st.number_input("Max steps", 1, 12, 5)
            creativity = st.slider("Creativity (temperature)", 0.0, 1.0, 0.2)
            start = st.form_submit_button("Run Agent 🚀")

        def plan_tasks(obj_text: str) -> List[str]:
            context_blurbs = []
            for u in st.session_state.uploads:
                context_blurbs.append(f"[FILE: {u['name']}]\n{(u.get('text') or '')[:2500]}")
            ctx = "\n\n".join(context_blurbs)

            plan_sys = (
                "You are a pragmatic project planner. Given a goal and optional context, "
                "return a concise JSON array of atomic steps. Each step should be actionable and less than 20 words."
            )
            user = (
                f"Goal: {obj_text}\n\nContext (may be truncated):\n{ctx}\n\nReturn JSON array only, no prose. Limit to {max_steps} steps."
            )
            llm = call_openrouter(
                messages=[{"role": "user", "content": user}],
                api_key=api_key,
                model=model,
                system_prompt=plan_sys,
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            steps = safe_parse_steps(llm.content, max_steps)
            _add_usage(llm.usage)
            return steps

        def execute_step(step: str, prior_artifacts: List[Dict[str, Any]]) -> Dict[str, Any]:
            parts = []
            if st.session_state.uploads:
                for u in st.session_state.uploads:
                    parts.append(f"[FILE: {u['name']}]\n{(u.get('text') or '')[:3500]}")
            if prior_artifacts:
                for i, art in enumerate(prior_artifacts, start=1):
                    preview = art.get("content", "")
                    parts.append(f"[ARTIFACT {i}: {art.get('title','step output')}]\n{preview[:3000]}")
            ctx = "\n\n".join(parts)

            exec_sys = (
                "You are an expert operator. Perform the requested step using the provided context. "
                "Return a high-quality, self-contained result. If the step is unclear, make reasonable assumptions and proceed."
            )
            user = f"Step to perform: {step}\n\nRelevant context (truncated):\n{ctx}"

            llm = call_openrouter(
                messages=[{"role": "user", "content": user}],
                api_key=api_key,
                model=model,
                system_prompt=exec_sys,
                temperature=creativity,
            )
            _add_usage(llm.usage)
            content = llm.content.strip()
            return {"title": step[:60], "content": content}

        def propose_next_moves(goal: str, artifacts: List[Dict[str, Any]]) -> List[str]:
            summ = "\n\n".join([f"[{a['title']}]\n{a['content'][:1200]}" for a in artifacts])
            sys_ = "You are a strategic advisor. Suggest concrete, high-leverage next moves (bulleted)."
            user = f"Objective: {goal}\n\nWhat we accomplished (truncated):\n{summ}\n\nSuggest 3-5 concise, high-impact next steps."
            llm = call_openrouter(
                messages=[{"role": "user", "content": user}],
                api_key=api_key,
                model=model,
                system_prompt=sys_,
                temperature=0.3,
            )
            _add_usage(llm.usage)
            lines = [ln.strip("- •* \t") for ln in llm.content.splitlines() if ln.strip()]
            return [ln for ln in lines if len(ln) > 3][:5]

        if start and goal.strip():
            with st.status("Agent running…", expanded=True) as status:
                st.write("Planning steps…")
                steps = plan_tasks(goal)
                if not steps:
                    st.error("Could not create a plan. Try rephrasing your goal.")
                else:
                    st.success(f"Planned {len(steps)} step(s).")
                    artifacts: List[Dict[str, Any]] = []
                    prog = st.progress(0.0, text="Executing steps…")
                    for i, step in enumerate(steps, start=1):
                        st.write(f"**Step {i}/{len(steps)}:** {step}")
                        art = execute_step(step, artifacts)
                        artifacts.append(art)
                        with st.expander(f"Output of step {i}"):
                            st.markdown(art["content"])  # render markdown
                        prog.progress(i / len(steps), text=f"Completed step {i}")
                        time.sleep(0.05)

                    st.write("Proposing next moves…")
                    ideas = propose_next_moves(goal, artifacts)
                    st.info("**Suggested next steps:**\n\n" + "\n".join([f"- {i}" for i in ideas]))

                    st.session_state.agent_runs.append({
                        "goal": goal,
                        "steps": steps,
                        "artifacts": artifacts,
                        "ideas": ideas,
                        "ts": time.time(),
                    })
                    status.update(state="complete", label="Agent complete ✅")

        if st.session_state.agent_runs:
            st.markdown("---")
            st.caption("Previous runs")
            for ridx, run in enumerate(reversed(st.session_state.agent_runs), start=1):
                with st.expander(f"Run {ridx}: {run['goal'][:60]}"):
                    st.markdown("**Plan**")
                    st.markdown("\n".join([f"{i+1}. {s}" for i, s in enumerate(run["steps"])]))
                    st.markdown("**Artifacts**")
                    for i, a in enumerate(run["artifacts"], start=1):
                        st.markdown(f"**{i}. {a['title']}**\n\n{a['content']}")
                    st.markdown("**Next moves**")
                    st.markdown("\n".join([f"- {i}" for i in run["ideas"]]))

    st.markdown("---")
    with st.container():
        u = st.session_state.usage
        st.caption(
            f"**Usage** – Requests: {u['total_requests']} | Prompt tokens: {u['total_prompt_tokens']} | Completion tokens: {u['total_completion_tokens']}"
        )

# -------------------------------------------------------------------------------------------------
# Tests (no network calls)
# -------------------------------------------------------------------------------------------------

class UtilTests(unittest.TestCase):
    def test_safe_parse_steps_json_object(self):
        raw = json.dumps({"steps": ["a", "b", "c"]})
        self.assertEqual(safe_parse_steps(raw, 5), ["a", "b", "c"])

    def test_safe_parse_steps_json_list(self):
        raw = json.dumps(["x", "y"])    
        self.assertEqual(safe_parse_steps(raw, 5), ["x", "y"]) 

    def test_safe_parse_steps_bullets(self):
        raw = "- first\n* second\n• third"
        self.assertEqual(safe_parse_steps(raw, 5), ["first", "second", "third"]) 

    def test_safe_parse_steps_empty(self):
        self.assertEqual(safe_parse_steps("", 5), [])

    def test_extract_text_from_bytes_txt(self):
        d = b"hello world"
        out = extract_text_from_bytes(d, "note.txt", "text/plain")
        self.assertEqual(out["type"], "text")
        self.assertIn("hello world", out["text"]) 

    def test_extract_text_from_bytes_csv_without_pandas(self):
        # Passes whether pandas is installed or not (fallback ok)
        d = b"a,b\n1,2\n3,4\n"
        out = extract_text_from_bytes(d, "t.csv")
        self.assertEqual(out["type"], "csv")
        self.assertIn("a,b", out["text"]) 

    def test_extract_text_from_bytes_binary_fallback(self):
        d = "☺".encode("utf-16", errors="ignore")
        out = extract_text_from_bytes(d, "bin.dat")
        self.assertEqual(out["type"], "binary")
        self.assertTrue(isinstance(out["text"], str))

    def test_run_cli_noop_without_api_key(self):
        # Should not raise SystemExit; should just print help and return
        args = argparse.Namespace(cli=True, run_tests=False, api_key=None, model=DEFAULT_MODEL,
                                  temperature=0.2, file=None, message=None, goal=None, max_steps=5)
        try:
            run_cli(args)
        except SystemExit as e:
            self.fail(f"run_cli unexpectedly exited with {e}")

    def test_run_cli_missing_api_key_with_message_no_exit(self):
        # Should not raise SystemExit even if message is provided
        args = argparse.Namespace(cli=True, run_tests=False, api_key=None, model=DEFAULT_MODEL,
                                  temperature=0.2, file=None, message="hi", goal=None, max_steps=5)
        try:
            run_cli(args)
        except SystemExit as e:
            self.fail(f"run_cli unexpectedly exited with {e}")

# -------------------------------------------------------------------------------------------------
# Entrypoint
# -------------------------------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AI Agent + Chat (Streamlit or CLI)")
    p.add_argument("--cli", action="store_true", help="Force CLI mode even if Streamlit is installed")
    p.add_argument("--run-tests", action="store_true", help="Run unit tests and exit")
    p.add_argument("--api-key", type=str, default=None, help="OpenRouter API key (or set OPENROUTER_API_KEY)")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model id")
    p.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    p.add_argument("--file", action="append", help="Path to context file (repeat for multiple)")
    p.add_argument("--message", type=str, default=None, help="Chat message (CLI mode)")
    p.add_argument("--goal", type=str, default=None, help="Agent objective (CLI mode)")
    p.add_argument("--max-steps", type=int, default=5, help="Max planning steps (CLI mode)")
    return p

if detect_streamlit_run():
    run_streamlit_app()
else:
    if __name__ == "__main__":
        parser = build_arg_parser()
        args = parser.parse_args()
        if args.run_tests:
            suite = unittest.defaultTestLoader.loadTestsFromTestCase(UtilTests)
            runner = unittest.TextTestRunner(verbosity=2)
            result = runner.run(suite)
            sys.exit(0 if result.wasSuccessful() else 1)
        if args.cli or not _HAS_STREAMLIT:
            run_cli(args)
        else:
            print(
                "Streamlit is installed, but you're not running under the Streamlit runtime.\n"
                "Use:  streamlit run app.py   (recommended)\n"
                "   or: python app.py --cli   (CLI mode)\n"
            )
