import argparse
import json
import sys

from . import config, core, models


def main():
    parser = argparse.ArgumentParser(prog="qbs", description="Question-Bottleneck Scoring")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("setup", help="choose the decision model")
    f = sub.add_parser("fit", help="answer candidate questions on labeled records, keep the best and fit their weights")
    f.add_argument("records", help="labeled records, .jsonl or .csv")
    f.add_argument("questions", help="candidate yes/no questions, one per line")
    f.add_argument("-o", "--out", default="model.json")
    f.add_argument("--label", default="is_bad")
    f.add_argument("--max-questions", type=int, default=10)
    s = sub.add_parser("score", help="score new records with a fitted model")
    s.add_argument("model", help="model.json from qbs fit")
    s.add_argument("records", help="records to score, .jsonl or .csv")
    s.add_argument("-o", "--out", help="write scores here instead of stdout")
    s.add_argument("--label", default="is_bad", help="dropped from records before scoring")
    args = parser.parse_args()

    if args.command == "setup":
        config.setup()
        return
    model = models.load(config.load())

    if args.command == "fit":
        rows = core.read_records(args.records)
        missing = sum(args.label not in r for r in rows)
        if missing:
            sys.exit(f"{missing} records have no {args.label} field")
        labels = [core.is_bad(r.pop(args.label)) for r in rows]
        if len(set(labels)) < 2:
            sys.exit(f"Every record has the same {args.label}; fitting needs both outcomes")
        questions = core.read_questions(args.questions)
        spec = core.fit(model, rows, labels, questions, args.records + ".answers.jsonl", args.max_questions)
        with open(args.out, "w") as out:
            json.dump(spec, out, indent=2)
        print(f"{spec['records']} records, {spec['bad']} bad")
        print(f"Kept {len(spec['questions'])} of {len(questions)} questions: CV AUC {spec['cv_auc']:.3f} "
              f"(all {len(questions)}: {spec['cv_auc_all_questions']:.3f})")
        for q, w in zip(spec["questions"], spec["weights"]):
            print(f"  {w:+.2f}  {q}")
        print(f"Saved {args.out}")

    if args.command == "score":
        with open(args.model) as f:
            spec = json.load(f)
        rows = core.read_records(args.records)
        for r in rows:
            r.pop(args.label, None)
        out = open(args.out, "w") if args.out else sys.stdout
        for row in core.score(model, spec, rows):
            out.write(json.dumps(row) + "\n")
