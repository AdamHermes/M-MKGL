"""
Quick analysis of the eval_details.jsonl produced by `--log-eval-details`.

Usage:
    python analyze_eval_log.py path/to/eval_details.jsonl
"""
import sys
import json
import pandas as pd


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def main(path):
    df = load(path)
    print(f"Loaded {len(df)} predictions ({df['task'].value_counts().to_dict()})")

    # Overall accuracy / rank stats, split by task direction
    print("\n=== Overall ===")
    print(df.groupby("task").agg(
        hits1=("correct", "mean"),
        mean_rank=("rank", "mean"),
        median_rank=("rank", "median"),
    ))

    # Worst-performing relations (lowest hits@1), tail-prediction only
    print("\n=== Hardest relations (tail_prediction, hits@1) ===")
    tail = df[df.task == "tail_prediction"]
    by_rel = tail.groupby("relation").agg(
        n=("correct", "size"),
        hits1=("correct", "mean"),
        mean_rank=("rank", "mean"),
    ).query("n >= 10").sort_values("hits1").head(15)
    print(by_rel)

    # A handful of confident-but-wrong examples: model was very sure, still wrong.
    print("\n=== Confidently wrong (high predicted_prob, incorrect) ===")
    wrong = df[~df.correct].sort_values("predicted_prob", ascending=False).head(10)
    for _, r in wrong.iterrows():
        print(f"[{r.task}] {r.query_entity} -[{r.relation}]-> ? "
              f"true={r.true_entity} pred={r.predicted_entity} "
              f"(p={r.predicted_prob:.3f}, rank={r.rank})")

    # Rank distribution buckets, handy for a quick "where does it fail" view
    print("\n=== Rank distribution ===")
    bins = [0, 1, 3, 10, 100, float("inf")]
    labels = ["rank=1", "rank<=3", "rank<=10", "rank<=100", "rank>100"]
    df["bucket"] = pd.cut(df["rank"], bins=bins, labels=labels)
    print(df["bucket"].value_counts().reindex(labels))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "eval_details.jsonl")