import json

path = r"c:\Users\bates\Desktop\Bipedal-imitation-rl\analyse_csv.ipynb"
with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    src = "".join(c.get("source", []))
    if "label='Config 2'" in src or "label='Config 3'" in src:
        print(i)
