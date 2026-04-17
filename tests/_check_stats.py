from stats_utils import welch_ttest, paired_bootstrap, mean_ci, pairwise_table
import numpy as np

rng = np.random.default_rng(0)
a = rng.normal(1.0, 0.5, 5)
b = rng.normal(0.5, 0.5, 5)
print("welch:", welch_ttest(a, b))
print("paired_boot:", paired_bootstrap(a, b))
print("mean_ci a:", mean_ci(a))
try:
    from scipy.stats import ttest_ind
    r = ttest_ind(a, b, equal_var=False)
    print("scipy welch:", float(r.statistic), float(r.pvalue))
except Exception as e:
    print("no scipy:", e)

data = {"A": a, "B": b, "C": rng.normal(0.8, 0.5, 5)}
print(pairwise_table(data, baseline="A", metric_name="demo"))
