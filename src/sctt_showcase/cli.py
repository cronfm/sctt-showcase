"""Run the local showcase server."""

import argparse

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="SCTT synthetic localization showcase")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: loopback only)")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run("sctt_showcase.api:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
