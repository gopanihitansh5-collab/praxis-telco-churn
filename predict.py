"""Read one strict customer JSON from a file or stdin; write prediction JSON."""

import argparse
import json
from pathlib import Path
import sys
from telco_churn.inference import predict_customer
from telco_churn.json_input import parse_customer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="-", help="JSON file, or - for stdin")
    parser.add_argument("--model", default="artifacts/model.joblib")
    args = parser.parse_args()
    try:
        text = sys.stdin.read() if args.input == "-" else Path(args.input).read_text()
        result = predict_customer(parse_customer(text), args.model)
    except (ValueError, OSError, KeyError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
