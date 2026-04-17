### Velocity-tracking demo -- per-config metrics with 95\% bootstrap CIs.

| config | metric | mean | 95% CI | n |
|---|---|---|---|---|
| alpha=0.25 | velocity MSE (m^2/s^2) | 0.004168 | [0.002679, 0.005845] | 41 |
| alpha=0.25 | success rate | 0.9756 | [0.9268, 1] | 41 |
| alpha=0.5 | velocity MSE (m^2/s^2) | 0.03155 | [0.01805, 0.04625] | 41 |
| alpha=0.5 | success rate | 0.9756 | [0.9268, 1] | 41 |
| no decay | velocity MSE (m^2/s^2) | 0.03252 | [0.02063, 0.04871] | 41 |
| no decay | success rate | 1 | [1, 1] | 41 |
| no imitation | velocity MSE (m^2/s^2) | 0.004285 | [0.002843, 0.005925] | 41 |
| no imitation | success rate | 1 | [1, 1] | 41 |

### Rotation / ramp demo -- per-config metrics with 95\% bootstrap CIs.

| config | metric | mean | 95% CI | n |
|---|---|---|---|---|
| alpha=0.25 | success rate | 0.8525 | [0.8249, 0.8802] | 651 |
| alpha=0.25 | max range (m) | nan | n/a | 0 |
| alpha=0.5 | success rate | 0.9616 | [0.9462, 0.9754] | 651 |
| alpha=0.5 | max range (m) | nan | n/a | 0 |
| no decay | success rate | 0.9232 | [0.9017, 0.9432] | 651 |
| no decay | max range (m) | nan | n/a | 0 |
| no imitation | success rate | 0.9508 | [0.9339, 0.9662] | 651 |
| no imitation | max range (m) | nan | n/a | 0 |

### Step-plane noisy-terrain demo -- per-config metrics with 95\% bootstrap CIs.

| config | metric | mean | 95% CI | n |
|---|---|---|---|---|
| alpha=0.25 | success rate | 0.4666 | [0.443, 0.4912] | 1596 |
| alpha=0.25 | max range (m) | 4.7 | [4.463, 4.937] | 1596 |
| alpha=0.5 | success rate | 0.5351 | [0.5115, 0.5587] | 1596 |
| alpha=0.5 | max range (m) | 4.741 | [4.496, 4.974] | 1596 |
| no decay | success rate | 0.4921 | [0.4676, 0.5161] | 1596 |
| no decay | max range (m) | 4.638 | [4.398, 4.872] | 1596 |
| no imitation | success rate | 0.3914 | [0.368, 0.4148] | 1596 |
| no imitation | max range (m) | 4.459 | [4.217, 4.692] | 1596 |

### Velocity-tracking (profile) demo -- per-config metrics with 95\% bootstrap CIs.

_(no data)_
