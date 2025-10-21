import argparse
import csv

EXPECTED_HEADER = ["Record Number","Category Id","Aspect Name","Aspect Value"]

def validate(path: str) -> bool:
    ok = True
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        try:
            header = next(reader)
        except StopIteration:
            print("Empty file")
            return False
        if header != EXPECTED_HEADER:
            print("Bad header:", header)
            ok = False
        for i, row in enumerate(reader, start=2):
            if len(row) != 4:
                print(f"Line {i}: expected 4 columns, got {len(row)}")
                ok = False
    return ok

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    args = parser.parse_args()
    res = validate(args.path)
    print("VALID" if res else "INVALID")

if __name__ == "__main__":
    main()
