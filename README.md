# Action Is All You Need

Small transformers trained from scratch on nothing but records of agents acting, and measured on one thing: do they
emit the right action. Working paper by Thiago Patzdorf and James (Genesis Innovation), 12 November 2026.

**Read the paper: [PAPER.md](PAPER.md).**

| Model | Parameters | Right tool | Right tool and all arguments | Chance |
|---|---|---|---|---|
| A-126 | 125.8 M | 88.2% | 55.4% | 31.5% |
| A-286 | 286.3 M | 90.8% | 63.1% | 31.5% |
| A-537 | 536.9 M | 94.0% | 68.5% | 31.5% |

Held-out trajectories that offer two or more tools; first tool call; greedy decoding; exact string match. Total GPU cost
of the three runs: about US$ 25 on one rented card.

## What this is, and what it is not

- A scaling experiment: same data, same recipe, three sizes. Every size improved every measure.
- The models are next-token predictors over agent trajectories. They are not chat models and were never instruction-tuned.
- The test is narrow. It scores the first action only, reads one of the two call formats present in the data, uses 314
  examples, and draws them from the same sources as training. Section 4 of the paper lists every limit.
- Research artifacts. Not evaluated for safety or for production use.

## What is in this repository

| Path | What |
|---|---|
| `PAPER.md` | The paper |
| `results/results.json` | The numbers in the table, machine-readable |
| `logs/` | Raw output of the scorer for each model |
| `scripts/regua_b.py` | The scorer ("régua") |
| `scripts/prep-acao.sh`, `scripts/baixar.sh` | Download and conversion of the corpus |
| `scripts/b4.sh`, `scripts/b5.sh` | Shuffle, training and scoring of the three models |
| `scripts/baixar2.sh`, `scripts/conv2.py`, `scripts/b6.sh` | The next run (larger corpus, larger model), in progress |
| `nanochat-commit.txt` | The commit of the nanochat codebase the runs used |

Script comments are in Portuguese. Paths assume a `/workspace` directory on the training machine.

## Data

Toucan-1.5M (Apache-2.0) and SWE-smith trajectories (MIT). Nothing proprietary was used.

## Weights

The three checkpoints and the tokenizer are being published on Hugging Face; the link will be added here.

## License

Code and weights: Apache-2.0 (see `LICENSE`). Paper text: CC BY 4.0.

## Citation

```
@misc{patzdorf2026action,
  title  = {Action Is All You Need},
  author = {Patzdorf, Thiago and James},
  year   = {2026},
  note   = {Genesis Innovation working paper}
}
```
