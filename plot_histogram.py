import csv
import matplotlib.pyplot as plt

probs = []

with open('probs.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        try:
            prob = float(row['probability'])
            probs.append(prob)
        except Exception:
            continue

if not probs:
    print("No probabilities found in probs.csv!")
    exit(1)

plt.figure(figsize=(8, 5))
plt.hist(probs, bins=20, color='skyblue', edgecolor='black')
plt.title('Model Probability Distribution')
plt.xlabel('Probability')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.tight_layout()
plt.savefig('probs_histogram.png')
plt.show() 