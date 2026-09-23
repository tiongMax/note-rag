"""Run the Note RAG API or Worker."""

import argparse

from note_rag.api.run import main_api, main_generation_worker, main_worker


def main() -> int:
    parser = argparse.ArgumentParser(description="Note RAG")
    parser.add_argument(
        "--mode",
        choices=["api", "worker", "generation-worker"],
        default="api",
        help="Run the API server or the background worker",
    )
    args = parser.parse_args()

    if args.mode == "api":
        return main_api()
    elif args.mode == "worker":
        return main_worker()
    elif args.mode == "generation-worker":
        return main_generation_worker()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
