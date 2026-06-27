import csv

# Read
with open('submission.csv', 'r', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

print(f"Before: rank 43={rows[42]['candidate_id']} rank 44={rows[43]['candidate_id']}")

# Sort: score descending, candidate_id ascending on ties
rows.sort(key=lambda r: (-float(r['score']), r['candidate_id']))

# Re-assign ranks
for i, r in enumerate(rows):
    r['rank'] = str(i + 1)

print(f"After:  rank 43={rows[42]['candidate_id']} rank 44={rows[43]['candidate_id']}")

# Write back
with open('submission.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['candidate_id', 'rank', 'score', 'reasoning'])
    writer.writeheader()
    writer.writerows(rows)

print("Done — run validate_submission.py submission.csv")
