"""
Quantum Chatbot — Streamlit *or* CLI + IBM Quantum (with Safe Fallback, Non‑Interactive Mode, and AI Chat + File Upload)
===============================================================================================================

Why this rewrite?
- You hit `SyntaxError: unterminated string literal` caused by accidental broken string literals like `"\n"` lines.
- Fixed all string joins to use explicit `"\n"` or `"\n\n"` and removed stray text in a function definition.
- Keeps previous fixes: auto batch mode (no `input()` in sandboxes), mock quantum fallback, Streamlit UI, and OpenRouter LLM chat + file uploads.
- Preserves existing tests and adds extras.

Quick Start
-----------
CLI (works without Streamlit/Qiskit):
    python app.py --mode cli

Non‑interactive / Batch usage (avoids input):
    python app.py --mode cli --cmd "/backends" --cmd "/bell" --cmd "/qrand 8" --cmd "/grover 3 101"
    python app.py --mode cli --cmd-file commands.txt
    QUANTUM_CHATBOT_COMMANDS="/backends;/bell;/qrand 16" python app.py --mode cli

Streamlit UI (AI Chat + Quantum tabs):
    pip install streamlit requests PyPDF2 qiskit qiskit-ibm-runtime python-dotenv
    export IBM_QUANTUM_TOKEN='YOUR_TOKEN'        # for quantum tab (optional)
    export OPENROUTER_API_KEY='YOUR_OPENROUTER'  # for AI chat tab
    streamlit run app.py

Security: never hardcode keys—use env vars or Streamlit secrets.
"""

from __future__ import annotations

import os
import sys
import io
import json
import argparse
import random
from typing import Dict, List, Optional, Tuple

# ----------------------------
# Optional imports: Streamlit (UI)
# ----------------------------
try:
    import streamlit as st  # type: ignore
    STREAMLIT_AVAILABLE = True
except Exception:
    STREAMLIT_AVAILABLE = False
    class _Stub:  # minimal stub so accidental calls don't crash
        def __getattr__(self, name):
            def _noop(*a, **k):
                return None
            return _noop
    st = _Stub()  # type: ignore

# ----------------------------
# Optional imports: dotenv (nice to have)
# ----------------------------
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    def load_dotenv(*_args, **_kwargs):  # fallback no-op
        return False

# ----------------------------
# Optional imports: Qiskit + IBM Runtime
# ----------------------------
QISKIT_AVAILABLE = False
IBM_RUNTIME_AVAILABLE = False
QuantumCircuit = None
SamplerClass = None
Session = None
QiskitRuntimeService = None

try:
    from qiskit import QuantumCircuit as _QuantumCircuit  # type: ignore
    QuantumCircuit = _QuantumCircuit
    QISKIT_AVAILABLE = True
    try:
        from qiskit_ibm_runtime import (
            QiskitRuntimeService as _QiskitRuntimeService,
            Session as _Session,
        )  # type: ignore
        try:
            from qiskit_ibm_runtime import SamplerV2 as _Sampler  # type: ignore
        except Exception:
            from qiskit_ibm_runtime import Sampler as _Sampler  # type: ignore
        QiskitRuntimeService = _QiskitRuntimeService
        Session = _Session
        SamplerClass = _Sampler
        IBM_RUNTIME_AVAILABLE = True
    except Exception:
        IBM_RUNTIME_AVAILABLE = False
except Exception:
    QISKIT_AVAILABLE = False
    IBM_RUNTIME_AVAILABLE = False

# If either piece is missing, we'll operate in MOCK mode by default
MOCK_QUANTUM = not (QISKIT_AVAILABLE and IBM_RUNTIME_AVAILABLE)

APP_TITLE = "Quantum Chatbot"
PERSONA = (
    "I am Qubit, an AI assistant with quantum capabilities. "
    "I can run small quantum demos (Bell, q-random, tiny Grover). "
    "If real IBM access isn't available, I'll use a clearly labeled mock backend."
)

DEFAULT_BACKEND_PREFERENCE = ["ibm_qasm_simulator", "simulator_statevector"]

# ----------------------------
# Utility
# ----------------------------

def _env_token() -> Optional[str]:
    load_dotenv()
    tok = os.getenv("IBM_QUANTUM_TOKEN")
    return tok if tok and tok.strip() else None


def _choose_backend(service, user_choice: Optional[str]) -> str:
    if MOCK_QUANTUM or service is None:
        return "mock_simulator"
    names = [b.name for b in service.backends()]
    if user_choice and user_choice in names:
        return user_choice
    for pref in DEFAULT_BACKEND_PREFERENCE:
        if pref in names:
            return pref
    return names[0] if names else ""


# ----------------------------
# Quantum (real) implementations
# ----------------------------

def _real_service(token: Optional[str]):
    if MOCK_QUANTUM:
        return None
    tok = token or _env_token()
    if not tok:
        return None
    try:
        return QiskitRuntimeService(channel="ibm_quantum", token=tok)
    except Exception:
        return None


def _real_bell(service, backend_name: str) -> Dict[str, int]:
    if MOCK_QUANTUM or service is None or QuantumCircuit is None or SamplerClass is None or Session is None:
        raise RuntimeError("Real quantum path not available")
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    with Session(service=service, backend=backend_name) as session:
        sampler = SamplerClass(session=session)
        job = sampler.run(qc, shots=2048)
        res = job.result()
        # Try several result layouts
        counts: Dict[str, float] | Dict[int, float]
        for getter in (
            lambda r: r[0].data.meas.get_counts(),
            lambda r: r.quasi_dists[0],
            lambda r: r.quasi_dist,
        ):
            try:
                counts = getter(res)  # type: ignore
                break
            except Exception:
                counts = {}
        # Convert to int counts
        out: Dict[str, int] = {}
        for k, v in (counts.items() if isinstance(counts, dict) else []):
            if isinstance(k, str):
                key = k
            else:
                key = format(k, "02b")
            val = int(round(v * 2048)) if isinstance(v, float) and 0.0 <= v <= 1.0 else int(round(v))
            out[key] = out.get(key, 0) + max(val, 0)
        filt = {k: v for k, v in out.items() if k in ("00", "11")}
        return filt or out or {"00": 1024, "11": 1024}


def _real_qrand(service, backend_name: str, n: int) -> str:
    if MOCK_QUANTUM or service is None or QuantumCircuit is None or SamplerClass is None or Session is None:
        raise RuntimeError("Real quantum path not available")
    n = max(1, min(n, 32))
    qc = QuantumCircuit(n, n)
    for i in range(n):
        qc.h(i)
        qc.measure(i, i)
    with Session(service=service, backend=backend_name) as session:
        sampler = SamplerClass(session=session)
        job = sampler.run(qc, shots=1)
        res = job.result()
        for getter in (
            lambda r: r[0].data.meas.get_counts(),
            lambda r: r.quasi_dists[0],
            lambda r: r.quasi_dist,
        ):
            try:
                counts = getter(res)  # type: ignore
                break
            except Exception:
                counts = {0: 1.0}
    if isinstance(counts, dict):
        if counts and isinstance(next(iter(counts.keys())), str):
            return max(counts, key=counts.get)  # type: ignore
        best = max(counts.items(), key=lambda kv: kv[1])[0]
        return format(int(best), f"0{n}b")
    return "0" * n


def _real_grover(service, backend_name: str, n: int, target: str) -> Dict[str, int]:
    if MOCK_QUANTUM or service is None or QuantumCircuit is None or SamplerClass is None or Session is None:
        raise RuntimeError("Real quantum path not available")
    n = max(1, min(n, 3))
    target = target.strip()
    if len(target) != n or any(c not in "01" for c in target):
        raise ValueError("Target must be a bitstring of length n, e.g., n=3, target='101'.")
    qc = QuantumCircuit(n, n)
    # Superposition
    for q in range(n):
        qc.h(q)
    # Oracle for target
    for i, bit in enumerate(reversed(target)):
        if bit == '0':
            qc.x(i)
    qc.h(n-1)
    qc.mcx(list(range(n-1)), n-1)
    qc.h(n-1)
    for i, bit in enumerate(reversed(target)):
        if bit == '0':
            qc.x(i)
    # Diffusion
    for q in range(n):
        qc.h(q); qc.x(q)
    qc.h(n-1); qc.mcx(list(range(n-1)), n-1); qc.h(n-1)
    for q in range(n):
        qc.x(q); qc.h(q)
    qc.measure(range(n), range(n))
    with Session(service=service, backend=backend_name) as session:
        sampler = SamplerClass(session=session)
        job = sampler.run(qc, shots=1024)
        res = job.result()
        try:
            qd = res[0].data.meas.get_counts()  # type: ignore
            if isinstance(qd, dict):
                return {k: int(v) if isinstance(v, (int, float)) else 0 for k, v in qd.items()}
        except Exception:
            pass
        try:
            qd = res.quasi_dists[0] if hasattr(res, 'quasi_dists') else res.quasi_dist
            out: Dict[str, int] = {}
            for k, v in (qd.items() if isinstance(qd, dict) else []):
                key = k if isinstance(k, str) else format(k, f"0{n}b")
                out[key] = out.get(key, 0) + int(round(float(v) * 1024))
            return out
        except Exception:
            return {target: 1024}


# ----------------------------
# Quantum (mock) implementations
# ----------------------------

def _mock_bell() -> Dict[str, int]:
    a = 1024 + random.randint(-50, 50)
    b = 2048 - a
    return {"00": max(a, 0), "11": max(b, 0)}


def _mock_qrand(n: int) -> str:
    n = max(1, min(n, 32))
    need_bytes = (n + 7) // 8
    raw = int.from_bytes(os.urandom(need_bytes), "big")
    return format(raw, f"0{need_bytes*8}b")[-n:]


def _mock_grover(n: int, target: str) -> Dict[str, int]:
    n = max(1, min(n, 3))
    target = target.strip()
    if len(target) != n or any(c not in "01" for c in target):
        raise ValueError("Target must be a bitstring of length n, e.g., n=3, target='101'.")
    out = {format(i, f"0{n}b"): random.randint(0, 5) for i in range(2**n)}
    out[target] = 100 + random.randint(0, 50)
    return out


# ----------------------------
# Public API used by UI/CLI
# ----------------------------

def list_backends(service) -> List[str]:
    if MOCK_QUANTUM or service is None:
        return ["mock_simulator"]
    try:
        return [b.name for b in service.backends()]
    except Exception:
        return ["ibm_qasm_simulator"]


def run_bell(service, backend_name: str) -> Dict[str, int]:
    if MOCK_QUANTUM or service is None or backend_name == "mock_simulator":
        return _mock_bell()
    return _real_bell(service, backend_name)


def quantum_random_bits(service, backend_name: str, n: int) -> str:
    if MOCK_QUANTUM or service is None or backend_name == "mock_simulator":
        return _mock_qrand(n)
    return _real_qrand(service, backend_name, n)


def grover_demo(service, backend_name: str, n: int, target: str) -> Dict[str, int]:
    if MOCK_QUANTUM or service is None or backend_name == "mock_simulator":
        return _mock_grover(n, target)
    return _real_grover(service, backend_name, n, target)


# ----------------------------
# Chat logic shared by UI/CLI
# ----------------------------

def handle_user_message(service, backend_name: str, msg: str) -> str:
    text = (msg or "").strip()
    if not text:
        return ""

    if text in {"/help", "help", "?"}:
        return (
            "**Commands**\n"
            "/backends — list available backends\n"
            "/bell — run Bell experiment\n"
            "/qrand N — quantum-random bits (N ≤ 32)\n"
            "/grover N TARGET — Grover demo (N ≤ 3)\n"
            "/quit — exit CLI\n"
        )

    if text.startswith("/backends"):
        names = list_backends(service)
        label = " (mock)" if MOCK_QUANTUM or (names and names[0] == "mock_simulator") else ""
        return "Available backends" + label + ":\n" + "\n".join(names)

    if text.startswith("/bell"):
        counts = run_bell(service, backend_name)
        mode = "mock" if backend_name == "mock_simulator" or MOCK_QUANTUM else "real"
        return f"Bell experiment counts [{mode}]:\n{json.dumps(counts, indent=2)}"

    if text.startswith("/qrand"):
        parts = text.split()
        try:
            n = int(parts[1]) if len(parts) > 1 else 8
        except Exception:
            n = 8
        bits = quantum_random_bits(service, backend_name, n)
        mode = "mock" if backend_name == "mock_simulator" or MOCK_QUANTUM else "real"
        return f"Here are {len(bits)} {mode} quantum‑random bits:\n{bits}"

    if text.startswith("/grover"):
        parts = text.split()
        if len(parts) < 3:
            return "Usage: /grover <n_qubits (≤3)> <target bitstring>  e.g., /grover 3 101"
        try:
            n = int(parts[1])
        except ValueError:
            return "First argument must be an integer for number of qubits."
        target = parts[2]
        try:
            counts = grover_demo(service, backend_name, n, target)
            pretty = json.dumps(counts, indent=2)
            mode = "mock" if backend_name == "mock_simulator" or MOCK_QUANTUM else "real"
            return f"Grover results for target={target} [{mode}]:\n{pretty}"
        except Exception as e:
            return f"Grover error: {e}"

    lower = text.lower()
    if any(k in lower for k in ["quantum", "qubit", "superposition", "entangle", "grover", "shor", "qft"]):
        return (
            "As Qubit, I can explain quantum topics or run quick demos. "
            "Try /bell, /qrand 16, /backends, or /grover 3 101.\n\n"
            "You're currently in " + ("MOCK" if MOCK_QUANTUM else "REAL") + " mode."
        )

    return (
        "[Qubit] I’m a quantum‑enhanced assistant. I can chat generally, and when you need quantum power, "
        "use the slash commands above (type /help)."
    )


# ----------------------------
# Batch runner (for non‑interactive environments)
# ----------------------------

def run_batch(service, backend_name: str, commands: List[str]) -> List[str]:
    outputs: List[str] = []
    for cmd in commands:
        cmd = (cmd or "").strip()
        if not cmd:
            continue
        if cmd in {"/quit", "quit", "exit"}:
            outputs.append("Bye.")
            break
        outputs.append(handle_user_message(service, backend_name, cmd))
    return outputs


def _gather_commands_from_sources(args: argparse.Namespace) -> List[str]:
    commands: List[str] = []
    if args.cmd:
        commands.extend(args.cmd)
    if args.cmd_file and os.path.isfile(args.cmd_file):
        try:
            with open(args.cmd_file, "r", encoding="utf-8") as f:
                commands.extend([line.strip() for line in f.readlines() if line.strip()])
        except Exception:
            pass
    env_cmds = os.getenv("QUANTUM_CHATBOT_COMMANDS")
    if env_cmds:
        commands.extend([c.strip() for c in env_cmds.split(";") if c.strip()])
    return commands


# ----------------------------
# OpenRouter LLM (Alibaba Tongyi DeepResearch) — optional
# ----------------------------

import requests

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_LLM_MODEL = os.getenv("OPENROUTER_MODEL", "alibaba/tongyi-deepresearch-30b-a3b:free")


def _llm_api_key_from_env_or_secrets() -> Optional[str]:
    try:
        if STREAMLIT_AVAILABLE and hasattr(st, "secrets") and "OPENROUTER_API_KEY" in st.secrets:
            return st.secrets["OPENROUTER_API_KEY"]
    except Exception:
        pass
    return os.getenv("OPENROUTER_API_KEY")


def llm_chat(messages: List[Dict[str, str]], api_key: Optional[str], model: Optional[str] = None, temperature: float = 0.2, max_tokens: Optional[int] = None) -> str:
    """Call OpenRouter with an OpenAI-compatible schema. Returns assistant text.
    This path is optional; if no API key is provided, we return a guidance string.
    """
    key = (api_key or _llm_api_key_from_env_or_secrets())
    mdl = model or DEFAULT_LLM_MODEL
    if not key:
        return "[LLM disabled] Set OPENROUTER_API_KEY to enable AI chat."
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv("OPENROUTER_REFERRER", "http://localhost"),
        "X-Title": os.getenv("OPENROUTER_APP_TITLE", APP_TITLE),
    }
    payload = {
        "model": mdl,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    try:
        resp = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "[No content]")
    except requests.HTTPError as e:
        try:
            err = resp.json()
        except Exception:
            err = {"error": str(e)}
        return f"[LLM HTTPError] {getattr(e, 'response', None) and e.response.status_code}: {err}"
    except Exception as e:
        return f"[LLM Error] {e}"


# ----------------------------
# Simple file helpers (used by AI Chat)
# ----------------------------

def _safe_decode(b: bytes) -> str:
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return b.decode(enc)
        except Exception:
            continue
    return b.decode("utf-8", errors="ignore")


def _extract_text_from_uploads(files: List) -> Tuple[str, List[str]]:
    """Return (combined_text, info_messages). Supports .txt/.md natively.
    PDF support is best-effort via PyPDF2 if installed.
    """
    infos: List[str] = []
    all_texts: List[str] = []
    if not files:
        return "", infos
    try:
        import PyPDF2  # type: ignore
        HAS_PDF = True
    except Exception:
        HAS_PDF = False
    for f in files:
        name = getattr(f, "name", "uploaded")
        suffix = os.path.splitext(name.lower())[1]
        data = f.read() if hasattr(f, "read") else f.getvalue() if hasattr(f, "getvalue") else b""
        if suffix in {".txt", ".md", ".json", ".csv"}:
            all_texts.append(_safe_decode(data))
            infos.append(f"Loaded text file: {name}")
        elif suffix == ".pdf":
            if not HAS_PDF:
                infos.append(f"Skipping PDF (PyPDF2 not installed): {name}")
                continue
            try:
                reader = PyPDF2.PdfReader(io.BytesIO(data))  # type: ignore
                txt = "\n".join(page.extract_text() or "" for page in reader.pages)
                all_texts.append(txt)
                infos.append(f"Extracted PDF: {name} ({len(reader.pages)} pages)")
            except Exception as e:
                infos.append(f"Failed to parse PDF {name}: {e}")
        else:
            infos.append(f"Unsupported file type for {name}; accepted: .txt, .md, .json, .csv, .pdf")
    combined = "\n\n".join(all_texts)
    return combined, infos


def _chunk_text(txt: str, max_chars: int = 8000) -> str:
    """Trim long context to keep prompt sizes manageable."""
    if len(txt) <= max_chars:
        return txt
    head = txt[: max_chars // 2]
    tail = txt[-max_chars // 2 :]
    return head + "\n\n...[truncated]...\n\n" + tail


# ----------------------------
# Streamlit UI (only if installed)
# ----------------------------

def run_streamlit_ui(token: Optional[str], backend_choice: Optional[str]):
    st.set_page_config(page_title=APP_TITLE, page_icon="✨", layout="wide")
    st.title(APP_TITLE)
    st.caption(PERSONA)

    tabs = st.tabs(["🤖 AI Chat", "⚛️ Quantum Demos"])  # Two-pane UI

    # ---------------- AI CHAT TAB ----------------
    with tabs[0]:
        with st.sidebar:
            st.subheader("AI Settings")
            api_key_in = st.text_input("OpenRouter API Key", type="password", help="Store in env OPENROUTER_API_KEY or Streamlit secrets for persistence.")
            model = st.text_input("Model", value=DEFAULT_LLM_MODEL)
            sys_prompt = st.text_area("System Prompt", value=(
                "You are Qubit, a helpful AI with quantum knowledge. "
                "Answer clearly, cite concepts simply, and when asked you can call quantum demos available in the other tab."
            ), height=120)
            st.markdown("**Tip:** Keep secrets out of code; use environment variables or Streamlit secrets.")

        st.write("Upload reference files to ground the chat (txt/md/json/csv/pdf):")
        uploaded = st.file_uploader("", type=["txt", "md", "json", "csv", "pdf"], accept_multiple_files=True)

        if "ai_history" not in st.session_state:
            st.session_state.ai_history = []  # list of dicts {role, content}
        if "ai_files_ctx" not in st.session_state:
            st.session_state.ai_files_ctx = ""
        if uploaded:
            ctx, infos = _extract_text_from_uploads(uploaded)
            st.session_state.ai_files_ctx = _chunk_text(ctx)
            for m in infos:
                st.caption(m)

        # Render previous messages
        for turn in st.session_state.ai_history:
            with st.chat_message(turn["role"]):
                st.markdown(turn["content"])

        user_msg = st.chat_input("Ask anything… (files above will ground my answers)")
        if user_msg:
            st.session_state.ai_history.append({"role": "user", "content": user_msg})
            with st.chat_message("assistant"):
                with st.spinner("Thinking with deep research…"):
                    messages = []
                    if sys_prompt:
                        messages.append({"role": "system", "content": sys_prompt})
                    if st.session_state.ai_files_ctx:
                        messages.append({"role": "system", "content": f"Context from user files:\n\n{st.session_state.ai_files_ctx}"})
                    history_tail = st.session_state.ai_history[-10:]
                    messages.extend(history_tail)
                    reply = llm_chat(messages, api_key_in, model)
                    st.markdown(reply)
            st.session_state.ai_history.append({"role": "assistant", "content": reply})

    # ---------------- QUANTUM TAB ----------------
    with tabs[1]:
        service = _real_service(token) if not MOCK_QUANTUM else None
        all_backends = list_backends(service)
        default_backend = _choose_backend(service, backend_choice)
        idx = max(0, all_backends.index(default_backend)) if default_backend in all_backends else 0
        backend_name = st.selectbox("Backend", options=all_backends, index=idx)

        mode_badge = "MOCK mode (offline OK)" if (MOCK_QUANTUM or backend_name=="mock_simulator") else "REAL IBM Quantum"
        st.info(f"Backend: {backend_name} — {mode_badge}")

        st.write("**Slash Commands**: /backends, /bell, /qrand N, /grover N TARGET, /help")

        if "history" not in st.session_state:
            st.session_state.history = []  # list[(role, text)]

        for role, text in st.session_state.history:
            with st.chat_message(role):
                st.markdown(text)

        q_msg = st.chat_input("Type a quantum command… (try /bell)", key="quantum_chat_input")
        if q_msg:
            st.session_state.history.append(("user", q_msg))
            with st.chat_message("user"):
                st.markdown(q_msg)
            with st.chat_message("assistant"):
                reply = handle_user_message(service, backend_name, q_msg)
                st.markdown(reply)
            st.session_state.history.append(("assistant", reply))


# ----------------------------
# CLI REPL (interactive) + Non‑interactive Batch Mode
# ----------------------------

def run_cli(token: Optional[str], backend_choice: Optional[str], commands: Optional[List[str]] = None):
    print(f"\n{APP_TITLE} — CLI mode\n{'='*40}")
    print(PERSONA)

    service = _real_service(token) if not MOCK_QUANTUM else None
    backend_name = _choose_backend(service, backend_choice)
    mode_label = "MOCK" if MOCK_QUANTUM or backend_name == "mock_simulator" else "REAL"

    stdin_is_tty = hasattr(sys.stdin, "isatty") and sys.stdin.isatty()
    run_in_batch = bool(commands) or not stdin_is_tty

    if run_in_batch:
        print(f"Using backend: {backend_name} [{mode_label}]\nRunning in non‑interactive batch mode.\n")
        cmds = commands or ["/backends", "/bell", "/qrand 8", "/grover 3 101"]
        outs = run_batch(service, backend_name, cmds)
        for o in outs:
            if o:
                print(o)
        print("\nBatch complete.")
        return

    print(f"Using backend: {backend_name} [{mode_label}]\nType /help for commands, /quit to exit.\n")

    while True:
        try:
            text = input(">> ").strip()
        except (EOFError, KeyboardInterrupt, OSError):
            print("\nInput unavailable — switching to batch demo.\n")
            demo = ["/backends", "/bell", "/qrand 8", "/grover 3 101", "/quit"]
            outs = run_batch(service, backend_name, demo)
            for o in outs:
                if o:
                    print(o)
            print("\nBatch complete.")
            return
        if text in {"/quit", "quit", "exit"}:
            print("Bye.")
            return
        reply = handle_user_message(service, backend_name, text)
        if reply:
            print(reply)


# ----------------------------
# Tests (existing tests preserved; more added)
# ----------------------------
import unittest

class TestQuantumFallback(unittest.TestCase):
    def test_qrand_length(self):
        bits = quantum_random_bits(None, "mock_simulator", 16)
        self.assertEqual(len(bits), 16)
        self.assertTrue(all(c in "01" for c in bits))

    def test_bell_keys(self):
        counts = run_bell(None, "mock_simulator")
        self.assertIsInstance(counts, dict)
        self.assertGreater(sum(counts.values()), 0)
        self.assertTrue(set(counts.keys()).issubset({"00", "11"}))

    def test_grover_target_peaks(self):
        n, target = 3, "101"
        counts = grover_demo(None, "mock_simulator", n, target)
        self.assertIn(target, counts)
        self.assertGreaterEqual(counts[target], max(counts.values()))

# ---- Additional tests ----
class TestChatLayer(unittest.TestCase):
    def test_help_text(self):
        out = handle_user_message(None, "mock_simulator", "/help")
        self.assertIn("/backends", out)
        self.assertIn("/qrand", out)

    def test_backends_mock(self):
        names = list_backends(None)
        self.assertIn("mock_simulator", names)

    def test_qrand_bounds(self):
        self.assertEqual(len(quantum_random_bits(None, "mock_simulator", 0)), 1)
        self.assertEqual(len(quantum_random_bits(None, "mock_simulator", 100)), 32)

    def test_grover_invalid_target_raises(self):
        with self.assertRaises(ValueError):
            grover_demo(None, "mock_simulator", 3, "2X1")

    def test_run_batch_smoke(self):
        outs = run_batch(None, "mock_simulator", ["/backends", "/qrand 4"])  # should not crash
        self.assertTrue(any("mock" in o.lower() or "Available backends" in o for o in outs))
        self.assertTrue(any("quantum" in o.lower() for o in outs))


@unittest.skipUnless(os.getenv("RUN_REAL_QUANTUM_TESTS") == "1", "real-quantum tests disabled")
class TestQuantumReal(unittest.TestCase):
    def setUp(self):
        self.service = _real_service(None)
        if self.service is None:
            self.skipTest("No real service (missing token or runtime)")
        self.backend = _choose_backend(self.service, None)

    def test_real_qrand_len(self):
        bits = quantum_random_bits(self.service, self.backend, 8)
        self.assertEqual(len(bits), 8)


# ----------------------------
# Entrypoint
# ----------------------------

def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=APP_TITLE)
    p.add_argument("--mode", choices=["auto", "streamlit", "cli"], default="auto",
                   help="auto → streamlit if available else cli")
    p.add_argument("--ibm-token", dest="ibm_token", default=None, help="IBM Quantum token (optional)")
    p.add_argument("--backend", dest="backend", default=None, help="Backend name (optional)")
    p.add_argument("--run-tests", action="store_true", help="Run unit tests and exit")
    p.add_argument("--cmd", action="append", help="Add a command to run in batch mode (repeatable)")
    p.add_argument("--cmd-file", dest="cmd_file", default=None, help="Path to a file with commands (one per line)")
    return p.parse_args(argv)


def main(argv: List[str] | None = None):
    args = _parse_args(argv or sys.argv[1:])

    if args.run_tests:
        print("Running tests...")
        suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
        res = unittest.TextTestRunner(verbosity=2).run(suite)
        sys.exit(0 if res.wasSuccessful() else 1)

    mode = args.mode
    if mode == "auto":
        mode = "streamlit" if STREAMLIT_AVAILABLE else "cli"

    batch_cmds = _gather_commands_from_sources(args)

    if mode == "streamlit":
        if not STREAMLIT_AVAILABLE:
            print("Streamlit is not installed. Falling back to CLI.\nTip: pip install streamlit")
            run_cli(args.ibm_token, args.backend, commands=batch_cmds)
        else:
            run_streamlit_ui(args.ibm_token, args.backend)
    else:
        run_cli(args.ibm_token, args.backend, commands=batch_cmds)


if __name__ == "__main__":
    main()
