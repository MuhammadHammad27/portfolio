"""Cross-platform development orchestrator.
Allows launching backend, frontend, or both concurrently with a single command.
"""
import sys
import subprocess
import argparse
import signal
import time

def run_backend():
    print("🚀 Starting FastAPI Backend at http://127.0.0.1:8000 ...")
    cmd = [sys.executable, "-m", "uvicorn", "backend.app.main:app", "--host", "127.0.0.1", "--port", "8000", "--reload"]
    return subprocess.Popen(cmd)

def run_frontend():
    print("🎨 Starting Streamlit Frontend at http://localhost:8501 ...")
    cmd = [sys.executable, "-m", "streamlit", "run", "frontend/app.py", "--server.port=8501"]
    return subprocess.Popen(cmd)

def run_tests():
    print("🧪 Executing automated test suite...")
    cmd = [sys.executable, "-m", "backend.tests.test_api"]
    res = subprocess.run(cmd)
    sys.exit(res.returncode)

def main():
    parser = argparse.ArgumentParser(description="Credit Risk Platform Launcher")
    parser.add_argument("--backend", action="store_true", help="Launch FastAPI backend only")
    parser.add_argument("--frontend", action="store_true", help="Launch Streamlit frontend only")
    parser.add_argument("--test", action="store_true", help="Run automated backend tests")
    args = parser.parse_args()

    if args.test:
        run_tests()
        return

    if args.backend and not args.frontend:
        p = run_backend()
        p.wait()
        return

    if args.frontend and not args.backend:
        p = run_frontend()
        p.wait()
        return

    # Default: launch both concurrently
    print("==================================================")
    print("💳 Credit Risk Assessment System Orchestrator")
    print("==================================================")
    backend_proc = run_backend()
    time.sleep(2)  # Give backend a moment to initialize
    frontend_proc = run_frontend()

    print("\n✓ Both services are running.")
    print("  • Backend API Docs:   http://127.0.0.1:8000/docs")
    print("  • Streamlit Web UI:   http://localhost:8501")
    print("\nPress Ctrl+C to terminate both services.\n")

    def signal_handler(sig, frame):
        print("\nStopping services...")
        backend_proc.terminate()
        frontend_proc.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        signal_handler(None, None)

if __name__ == "__main__":
    main()
