
import csv
import random

LOW = 0
HIGH = 1000
NUM_ROWS = 10000
NOISE_RATE = 0.05  

random.seed(42) 

def build_row():
    a = random.randint(LOW, HIGH)
    b = random.randint(LOW, HIGH)

    if random.random() < NOISE_RATE:
        total = random.randint(LOW, HIGH * 2) 
    else:
        total = a + b

    return a, b, total


def main():
    noisy = 0
    with open("dataset.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["a", "b", "total"]) 
        for _ in range(NUM_ROWS):
            a, b, total = build_row()
            if total != a + b:
                noisy += 1
            writer.writerow([a, b, total])

    print(f"Wrote {NUM_ROWS} rows to dataset.csv")
    print(f"Noisy rows: {noisy} ({noisy / NUM_ROWS:.1%})")


if __name__ == "__main__":
    main()
