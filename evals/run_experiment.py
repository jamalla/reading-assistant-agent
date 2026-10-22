"""Run the eval dataset through the agent with one prompt version, scored in Langfuse.

Usage:  python evals/run_experiment.py --prompt v1
Then open Langfuse > Datasets > reading-assistant-evals to compare runs.
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from langfuse import get_client  # noqa: E402

import agent  # noqa: E402
from evaluators import JUDGE_MODEL, averages, fits_request, real_books  # noqa: E402
from upload_dataset import DATASET  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="v1", choices=sorted(agent.PROMPTS))
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    langfuse = get_client()
    dataset = langfuse.get_dataset(DATASET)
    the_agent = agent.build_agent(args.prompt)

    def task(*, item, **kwargs):
        return agent.ask(the_agent, item.input["question"], callbacks=agent.langfuse_callbacks())

    result = dataset.run_experiment(
        name=f"prompt-{args.prompt}",
        run_name=f"prompt-{args.prompt} {datetime.now():%Y-%m-%d %H:%M}",
        description=f"Agent with system prompt {args.prompt}",
        task=task,
        evaluators=[real_books, fits_request],
        run_evaluators=[averages],
        max_concurrency=args.concurrency,
        metadata={"prompt": args.prompt, "agent_model": agent.MODEL_ID, "judge_model": JUDGE_MODEL},
    )
    langfuse.flush()
    print(result.format())


if __name__ == "__main__":
    main()
