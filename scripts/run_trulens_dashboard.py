from __future__ import annotations

import argparse
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the TruLens tracing dashboard.")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--host", default="localhost")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = [
        sys.executable,
        "-m",
        "trulens.dashboard.run",
        "--port",
        str(args.port),
        "--address",
        args.host,
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        print("TruLens dashboard failed to start.", file=sys.stderr)
        print(
            "Verify that TruLens dashboard dependencies are installed and compatible, "
            "then retry this command:",
            file=sys.stderr,
        )
        print(" ".join(command), file=sys.stderr)
        raise SystemExit(exc.returncode) from exc
    except FileNotFoundError as exc:
        print("Python executable was not found while launching TruLens dashboard.", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
