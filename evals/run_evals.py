"""
Evaluation framework for the commerce AI agent.

Runs structured experiments via Arize Phoenix to measure:
  1. Routing accuracy    — did the agent call the correct tool?
  2. Parameter quality   — did the agent extract reasonable parameters?
  3. Category relevance  — do returned products match expected categories?
  4. Colour relevance    — do returned products match expected colours?
  5. Convergence score   — did the agent take the optimal path?

Usage:
    python evals/run_evals.py
    python evals/run_evals.py --dry-run          # test on 1 example
    python evals/run_evals.py --name "v2-prompt"  # name the experiment
"""

import json
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import phoenix as px
from phoenix.client import Client
from phoenix.client.experiments import run_experiment, evaluate_experiment

# Load test cases

EVALS_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(EVALS_DIR, "test_cases.json")) as f:
    test_cases = json.load(f)


# Task: run the agent on each example
# collect results during the first run for convergence analysis after the experiment completes
# Module-level collector
_experiment_results = []

def agent_task(example):
    from agent.strands_agent import run_agent

    query = example.input["query"]
    image = example.input.get("image")

    try:
        result = run_agent(query, image)
        if isinstance(result, dict):
            output = {
                "tool_called": result.get("tool_name"),
                "tool_params": result.get("tool_params"),
                "products": result.get("products", []),
                "response": result.get("text", ""),
                "num_turns": result.get("num_turns", 1),
            }
        else:
            output = {
                "tool_called": None,
                "tool_params": None,
                "products": [],
                "response": str(result),
                "num_turns": 1,
            }
    except Exception as e:
        output = {
            "error": str(e),
            "tool_called": None,
            "tool_params": None,
            "products": [],
            "response": "",
            "num_turns": 0,
        }

    _experiment_results.append(output)
    return output

# Evaluators

def routing_accuracy(output, expected) -> bool:
    """Did the agent call the correct tool (or correctly call no tool)?"""
    return output.get("tool_called") == expected.get("expected_tool")


def parameter_quality(output, expected) -> bool:
    """
    Did the agent extract a reasonable query parameter?
    For product_recommendation: the extracted query should contain
    at least one keyword from the original query.
    For no-tool calls: always passes.
    """
    expected_tool = expected.get("expected_tool")
    if expected_tool is None:
        return True  # no tool expected, nothing to check

    params = output.get("tool_params")
    if params is None:
        return False  # tool was expected but no params extracted

    if expected_tool == "product_recommendation":
        extracted_query = params.get("query", "").lower()
        # Check if key terms from original query appear in extracted query
        original_words = expected.get("_original_query", "").lower().split()
        # Filter out stop words
        stop_words = {"for", "me", "a", "an", "the", "in", "to", "find", "show", "recommend"}
        content_words = [w for w in original_words if w not in stop_words]
        if not content_words:
            return True
        matches = sum(1 for w in content_words if w in extracted_query)
        return matches >= 1

    return True  # image search doesn't have meaningful params to check


def category_relevance(output, expected) -> float:
    """
    What fraction of returned products match the expected categories?
    Returns 1.0 for general chat (no products expected).
    """
    expected_cats = expected.get("expected_categories", [])
    if not expected_cats:
        return 1.0  # no expectation — passes

    products = output.get("products", [])
    if not products:
        return 0.0  # expected products but got none

    matched = sum(
        1 for p in products
        if any(cat.lower() in (p.get("category") or "").lower() for cat in expected_cats)
    )
    return matched / len(products)


def colour_relevance(output, expected) -> float:
    """
    What fraction of returned products match the expected colours?
    Returns 1.0 when no colour expectation is set.
    """
    expected_colours = expected.get("expected_colours", [])
    if not expected_colours:
        return 1.0

    products = output.get("products", [])
    if not products:
        return 0.0

    matched = sum(
        1 for p in products
        if any(c.lower() in (p.get("colour") or "").lower() for c in expected_colours)
    )
    return matched / len(products)


# Remove the old convergence_score evaluator from the evaluators list
# and compute it after the experiment runs

def compute_convergence(experiment_results):
    """
    Convergence score per the formal definition:
    1. Group runs by query type (chat vs tool-calling)
    2. S_optimal = min steps observed within each group
    3. Score = (1/N) * sum(min(1, S_optimal / S_agent_i))
    """
    # Separate into groups of similar queries
    groups = {"chat": [], "tool": []}

    for run in experiment_results:
        output = run.get("output", {})
        expected = run.get("expected", {})
        num_turns = output.get("num_turns", 1) if output else 1

        if expected.get("expected_tool") is None:
            groups["chat"].append(num_turns)
        else:
            groups["tool"].append(num_turns)

    # Calculate per-group S_optimal
    s_optimal = {}
    for group_name, turns_list in groups.items():
        if turns_list:
            s_optimal[group_name] = min(turns_list)

    # Calculate overall convergence score
    all_scores = []
    for run in experiment_results:
        output = run.get("output", {})
        expected = run.get("expected", {})
        s_agent_i = output.get("num_turns", 1) if output else 1

        if s_agent_i == 0:
            all_scores.append(0.0)
            continue

        group = "chat" if expected.get("expected_tool") is None else "tool"
        opt = s_optimal.get(group, 1)

        all_scores.append(min(1.0, opt / s_agent_i))

    N = len(all_scores)
    overall = sum(all_scores) / N if N > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"CONVERGENCE ANALYSIS")
    print(f"{'='*60}")
    print(f"Chat queries:  S_optimal = {s_optimal.get('chat', 'N/A')} steps, N = {len(groups['chat'])}")
    print(f"Tool queries:  S_optimal = {s_optimal.get('tool', 'N/A')} steps, N = {len(groups['tool'])}")
    print(f"Overall Convergence Score: {overall:.4f}")
    print(f"{'='*60}")

    return overall


def no_error(output) -> bool:
    """Did the agent complete without errors?"""
    return not bool(output.get("error"))


# Main
# Runs the agent once, collects results during the experiment, then computes convergence as an aggregate metric at the end.
def main():
    parser = argparse.ArgumentParser(description="Run agent evaluations")
    parser.add_argument("--name", default="baseline", help="Experiment name")
    parser.add_argument("--dry-run", action="store_true", help="Test on 1 example")
    args = parser.parse_args()

    print(f"Loading {len(test_cases)} test cases...")

    # Inject original query into expected output for parameter_quality evaluator
    outputs = []
    for tc in test_cases:
        out = {
            "expected_tool": tc["expected_tool"],
            "expected_categories": tc["expected_categories"],
            "expected_colours": tc["expected_colours"],
            "_original_query": tc["query"],
        }
        outputs.append(out)

    # Upload dataset to Phoenix
    px_client = Client()

    # Check if dataset already exists, create if not
    dataset_name = "commerce-agent-eval"
    try:
        dataset = px_client.datasets.get_dataset(dataset=dataset_name)
        print(f"Using existing dataset: {dataset_name}")
    except Exception:
        dataset = px_client.datasets.create_dataset(
            name=dataset_name,
            inputs=[{"query": tc["query"], "image": tc["image"]} for tc in test_cases],
            outputs=outputs,
        )
        print(f"Created dataset: {dataset_name}")

    # Run experiment
    print(f"\nRunning experiment: {args.name}")
    print("=" * 60)

    experiment = run_experiment(
        dataset=dataset,
        task=agent_task,
        evaluators=[
            no_error,
            routing_accuracy,
            parameter_quality,
            category_relevance,
            colour_relevance,
        ],
        experiment_name=args.name,
        dry_run=args.dry_run,
    )

    # Compute convergence from collected results
    if not args.dry_run and _experiment_results:
        paired = [
            {"output": _experiment_results[i], "expected": outputs[i]}
            for i in range(len(_experiment_results))
        ]
        compute_convergence(paired)

    print("\n" + "=" * 60)
    print(f"Experiment '{args.name}' complete.")
    print("View results in Phoenix UI.")


if __name__ == "__main__":
    main()