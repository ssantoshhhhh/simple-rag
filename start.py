#!/usr/bin/env python3
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3:instruct"
BACKENDS = ("auto", "ollama", "openrouter")
EXAMPLES = {
    "ollama": ROOT / "examples" / "simple_rag_bm25_ollama.py",
    "openrouter": ROOT / "examples" / "simple_rag_bm25_openrouter.py",
}
UI_EXAMPLE = ROOT / "examples" / "simple_rag_bm25_ollama_ui.py"
REQUIRED_MODULES = ("rag", "pdftotext", "rank_bm25", "sentence_transformers", "requests", "dotenv")


def venv_python() -> Path:
    if os.name == "nt":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def reexec_in_venv() -> None:
    if os.environ.get("SIMPLE_RAG_VENV") == "1":
        return
    python = venv_python()
    if not python.exists():
        return
    if Path(sys.prefix).resolve() == VENV.resolve():
        return
    print(f"Using virtual environment: {VENV}")
    os.environ["SIMPLE_RAG_VENV"] = "1"
    os.execv(str(python), [str(python), str(Path(__file__).resolve()), *sys.argv[1:]])


def read_dotenv() -> dict[str, str]:
    values: dict[str, str] = {}
    path = ROOT / ".env"
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def check_python() -> None:
    required = (3, 10)
    if sys.version_info[:2] < required:
        sys.exit(f"Python {required[0]}.{required[1]}+ is required, found {sys.version.split()[0]}.")
    if sys.version_info >= (3, 14):
        print("Warning: pinned dependencies (torch, sentence-transformers) may not support Python 3.14.")


def missing_modules() -> list[str]:
    return [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]


def install_dependencies() -> None:
    print("Installing Python dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], cwd=ROOT)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."], cwd=ROOT)


def ollama_models() -> list[str] | None:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return None
    return [str(model.get("name", "")) for model in payload.get("models", [])]


def has_openrouter_key(values: dict[str, str]) -> bool:
    return bool(os.getenv("OPENROUTER_API_KEY") or values.get("OPENROUTER_API_KEY"))


def select_backend(requested: str, dotenv: dict[str, str]) -> str:
    if requested != "auto":
        return requested
    if ollama_models() is not None:
        return "ollama"
    if has_openrouter_key(dotenv):
        return "openrouter"
    sys.exit(
        "No LLM backend available.\n"
        "  Ollama: run `brew install ollama && brew services start ollama && ollama pull llama3:instruct`\n"
        "  OpenRouter: set OPENROUTER_API_KEY in .env (see .env.example)"
    )


def prepare_ollama(force: bool) -> None:
    models = ollama_models()
    if models is None:
        sys.exit(
            f"Ollama is not reachable at {OLLAMA_URL}.\n"
            "Start it with `brew services start ollama` (macOS) or `ollama serve`."
        )
    if OLLAMA_MODEL not in models and not any("llama3" in name for name in models):
        message = f"Ollama model `{OLLAMA_MODEL}` is not installed. Run `ollama pull {OLLAMA_MODEL}`."
        if not force:
            sys.exit(message)
        print(f"Warning: {message}")


def prepare_openrouter(dotenv: dict[str, str]) -> None:
    if not has_openrouter_key(dotenv):
        sys.exit(
            "OPENROUTER_API_KEY is not set.\n"
            "Add it to .env in the project root (see .env.example)."
        )


def run_cli(backend: str, env: dict[str, str]) -> int:
    script = EXAMPLES[backend]
    if not script.exists():
        sys.exit(f"Example script not found: {script}")
    print(f"Starting {script.name} (backend: {backend})", flush=True)
    return subprocess.call([sys.executable, str(script)], cwd=ROOT, env=env)


def run_ui(env: dict[str, str], install: bool) -> int:
    if importlib.util.find_spec("streamlit") is None:
        if not install:
            sys.exit("Streamlit is not installed. Run `pip install streamlit` or `python start.py --ui --install`.")
        print("Installing Streamlit...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"], cwd=ROOT)
    print(f"Starting Streamlit UI: {UI_EXAMPLE.name}", flush=True)
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(UI_EXAMPLE)], cwd=ROOT, env=env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check prerequisites and start the simple-rag project.")
    parser.add_argument("--backend", choices=BACKENDS, default="auto", help="LLM backend to use (default: auto-detect)")
    parser.add_argument("--ui", action="store_true", help="launch the Streamlit web UI instead of the CLI")
    parser.add_argument("--install", action="store_true", help="install missing Python dependencies automatically")
    parser.add_argument("--force", action="store_true", help="continue even if the Ollama model is missing")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reexec_in_venv()
    check_python()

    if not VENV.exists():
        print(f"Warning: virtual environment not found at {VENV}. See GETTING_STARTED.md.")

    missing = missing_modules()
    if missing:
        if not args.install:
            sys.exit(f"Missing Python modules: {', '.join(missing)}\nRun `python start.py --install` or see GETTING_STARTED.md.")
        install_dependencies()

    dotenv = read_dotenv()
    backend = select_backend(args.backend, dotenv)
    if backend == "ollama":
        prepare_ollama(args.force)
    else:
        prepare_openrouter(dotenv)
    if args.ui and backend != "ollama":
        sys.exit("The Streamlit UI currently supports Ollama only. Use `--backend ollama` or drop `--ui`.")

    env = {**os.environ, **dotenv}
    return run_ui(env, args.install) if args.ui else run_cli(backend, env)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print()
        sys.exit(130)
