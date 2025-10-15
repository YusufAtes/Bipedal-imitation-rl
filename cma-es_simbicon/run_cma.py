import subprocess
import numpy as np
# Starting value
tv_list = np.linspace(0.1, 2.1, 41)
total_runs = len(tv_list)
# How many runs you want
first_run = True
i = 0
for tv in range(1):
    tv = 1.715
    i += 1
    lm = "walk"
    lp = "/home/baran/Bipedal-imitation-rl/locomotion-master(1)/locomotion-master/settings/cma_config_1.76.yml"

    if first_run:
        first_run = False
        n_gen = 10
        if lm == "walk":
            # lp = "settings/config.yml"
            pass
    else:
        n_gen = 10
        lp = f"settings/cma_config_{prev_tv:.2f}.yml"

    # print(f"tv = {tv:.2f}")
    subprocess.run([
        f"python", "cma_simbicon.py",
        "-lp", lp,
        "-lm", lm,
        "-sm", f"{lm}",
        "-sp", f"settings/cma_config_{tv:.2f}.yml",
        "-tv", str(tv),
        "-ng", str(n_gen)
    ])
    prev_tv = tv
    print(f"Completed {i} out of {total_runs} runs")