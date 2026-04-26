"""Entry point: `python -m ktt_mcp`."""

import argparse
import logging
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(prog="ktt-mcp", description="KTT MCP server (stdio).")
    parser.add_argument("--workdir", default=None, help="Run directory root (default: ./.ktt-mcp).")
    parser.add_argument("--log-file", default=None, help="Mirror logs to this file.")
    args = parser.parse_args()

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(
        level=logging.DEBUG if os.environ.get("KTT_MCP_DEBUG") else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )

    from ktt_mcp.server import build_server

    server = build_server(workdir=args.workdir)
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
