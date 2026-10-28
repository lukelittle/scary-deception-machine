#!/usr/bin/env python3
"""Launch the Deception Machine dashboard."""

import subprocess
import sys
import os

if __name__ == "__main__":
    dashboard_path = os.path.join(
        os.path.dirname(__file__),
        "src",
        "deception_machine",
        "dashboard",
        "app.py"
    )

    port = sys.argv[1] if len(sys.argv) > 1 else "8501"

    print("Launching The Deception Machine Dashboard...")
    print(f"URL: http://localhost:{port}")
    print()

    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        dashboard_path,
        "--server.port", port,
    ])
