import argparse, csv
from pathlib import Path
from collections import Counter

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels", required=True, help="Folder containing YOLO .txt labels")
    p.add_argument("--output", default="outputs/class_balance.csv")
    a = p.parse_args()
    counts = Counter()
    files = list(Path(a.labels).glob("*.txt"))
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if parts:
                try: counts[int(float(parts[0]))] += 1
                except ValueError: pass
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    with open(a.output, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp); writer.writerow(["class_id", "instances"])
        writer.writerows(sorted(counts.items()))
    total = sum(counts.values())
    print(f"Label files: {len(files)} | instances: {total}")
    for cls, n in sorted(counts.items()):
        print(f"class {cls}: {n} ({100*n/max(total,1):.2f}%)")
    print("Saved:", a.output)
if __name__ == "__main__": main()
