"""Read a raw customer profile from a JSON file or stdin and emit JSON."""
import argparse
import json
import sys
from telco_churn.inference import predict_customer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='-', help='JSON file, or - for stdin')
    parser.add_argument('--model', default='artifacts/model.joblib')
    args = parser.parse_args()
    try:
        payload = json.load(sys.stdin) if args.input == '-' else json.loads(open(args.input).read())
        result = predict_customer(payload, args.model)
    except (ValueError, OSError, KeyError) as exc:
        print(json.dumps({'error': str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(result))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
