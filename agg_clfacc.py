import glob
import re
import os
import numpy as np
import pandas as pd
from collections import defaultdict

acc = defaultdict(list)

for f in glob.glob("./results_our/*_clfacc_*.npy"):
    m = re.search(r"/([^/]+)_(state|transformer|cnn|linear)_clfacc_(\d+)_(\d+)\.npy$", f)
    if m is None:
        print(f"[skip] unmatched file: {f}")
        continue

    data, model = m.group(1), m.group(2)
    acc[(data, model)].append(float(np.load(f)))

os.makedirs("./results_table", exist_ok=True)

rows = []
lines = []

for (data, model), v in sorted(acc.items()):
    mean = float(np.mean(v))
    std = float(np.std(v))
    n = len(v)

    line = f"{data:10s} {model:12s} acc={mean:.4f} ± {std:.4f} (n={n})"
    print(line)
    lines.append(line)

    rows.append({
        "data": data,
        "model": model,
        "acc_mean": mean,
        "acc_std": std,
        "n": n,
    })

# txt 저장
with open("./results_table/clfacc_summary.txt", "w") as fp:
    fp.write("\n".join(lines) + "\n")

# csv 저장
df = pd.DataFrame(rows)
df.to_csv("./results_table/clfacc_summary.csv", index=False)

print("\n[saved] ./results_table/clfacc_summary.txt")
print("[saved] ./results_table/clfacc_summary.csv")