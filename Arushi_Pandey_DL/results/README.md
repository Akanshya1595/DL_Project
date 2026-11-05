# Arushi Pandey: results

Split fingerprint: `591155c8c8bd` | GPU: Tesla T4 | seed 42

| Model | Test acc | Macro-P | Macro-R | Macro-F1 | 4-stream acc | Params (M) | Train (min) | Latency (ms) | Best val F1 (epoch) |
|---|---|---|---|---|---|---|---|---|---|
| Scratch CNN | 0.8776 | 0.8466 | 0.8688 | 0.8567 | 0.9240 | 2.36 | 5.7 | 1.77 | 0.8549 (30) |
| Scratch CNN (no augmentation) | 0.8802 | 0.8559 | 0.8578 | 0.8565 | 0.9227 | 2.36 | 4.9 | 1.85 | 0.8518 (29) |

Augmentation ablation: macro-F1 with augmentation minus without = **+0.0002**.

Worst classes by test F1: Scratch CNN: metal (0.78), plastic (0.79), white-glass (0.83); Scratch CNN (no augmentation): plastic (0.79), metal (0.80), battery (0.82)
