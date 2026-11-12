# Action Is All You Need

**Thiago Patzdorf and James — Genesis Innovation. Working paper, 12 November 2026.**
**Weights, scorer and training scripts are released with this paper (Apache-2.0); the text is CC BY 4.0.**

## Abstract

Language models are trained to talk and then taught, late and expensively, to act. We ask what happens if the order is
reversed: train a small transformer from scratch on nothing but records of agents acting (tool definitions, tool calls,
tool results, and whatever text sits between them) and measure only one thing, whether it emits the right action.
We trained three models of 126 M, 286 M and 537 M parameters on 1.53 million open agent trajectories, on one rented GPU,
for a total of about US$ 25. On held-out trajectories with two or more tools available, the largest model picks the
right tool 94.0% of the time and writes the tool name and every argument exactly right 68.5% of the time, against a
chance level of 31.5%. Every increase in size improved every measure. The models do not chat and were never asked to.
This is an early result on a narrow test, and we say exactly how narrow below.

## 1. The claim

"Attention Is All You Need" (Vaswani et al., 2017) removed recurrence and kept one mechanism. We make a claim of the same
shape about the training signal, not the architecture: for a system whose job is to operate software, the action
is the unit worth predicting. Fluent prose is a by-product that a system can acquire later, or borrow.

Three observations led here.

1. **What we pay for in production is action.** Our agents run inside client businesses. A turn is useful when a tool
   was called with the right arguments and the state of the world changed. In our own logs, a model that says "sent" without
   calling the tool is the most expensive failure we have.
2. **General models are graded on the wrong thing for this job.** On the public multi-turn function-calling test we ran
   (BFCL, multi-turn base, 200 tasks), four current production models scored between 53.5% and 70%. This was our own run through
   a compatibility bridge, not an official leaderboard configuration, so we do not name the models. They are far larger than
   anything in this paper and most of their capacity is spent on things an operator does not need.
3. **The bitter lesson (Sutton, 2019).** We did not design a planner, a parser for instructions or a tool-selection
   module. We did not use our own data, which is small. We took the largest open corpus of actions we could find,
   a standard architecture, and let scale decide.

## 2. Data

| Source | License | What it is |
|---|---|---|
| Toucan-1.5M (Kimi-K2, OSS and Qwen3 subsets) | Apache-2.0 | Tool-use trajectories against real MCP servers |
| SWE-smith trajectories | MIT | Software-engineering agent trajectories |

- 1,527,259 documents after conversion. Each document is the list of available tools followed by the full message
  sequence, including the natural-language turns. We did not strip the prose.
- Documents were shuffled with a fixed seed (20261111) and written to 152 shards; the last shard is held out.
- A 32,768-token BPE vocabulary was trained on this corpus. The corpus is roughly 2.75 billion tokens under it
  (estimated from the point at which the largest run wrapped into a second pass).

**A mistake worth recording.** Our first series was trained on shards in directory order. The two model sizes saw
different sources with different call formats, and the validation shard held a third. The larger model scored 57% on
tool name and 0% on arguments. Size and data source had changed together, so the comparison meant nothing. Every number
in this paper is from the second series, on shuffled data.

## 3. Models and training

Decoder-only transformers from the open nanochat codebase, unchanged except for the data. Context length 2,048 tokens.
Training tokens follow the codebase's default of 12 tokens per matrix parameter.

| Model | Layers | Width | Parameters | Training tokens | Wall time | GPU cost |
|---|---|---|---|---|---|---|
| A-126 | 8 | 512 | 125.8 M | 0.50 B | 32 min | under US$ 1 |
| A-286 | 12 | 768 | 286.3 M | 1.32 B | 2 h 57 min | about US$ 5 |
| A-537 | 16 | 1,024 | 536.9 M | 2.82 B | 11 h 27 min | about US$ 18 |

One RTX PRO 6000 (96 GB), rented at US$ 1.61 per hour. Peak memory for the largest run was 48 GB. A-537 passed over
the corpus slightly more than once.

## 4. Evaluation

For each held-out trajectory we give the model everything up to the start of the first tool call and let it write,
greedily (temperature 0), the tool name and the arguments. We compare with the recorded call by exact string match.
There is no judge model. Chance is the mean of one over the number of tools offered in each example.

| Model | Validation loss (bits per byte) | Right tool | Right tool and all arguments | Chance |
|---|---|---|---|---|
| A-126 | 0.125 | 88.2% | 55.4% | 31.5% |
| A-286 | 0.099 | 90.8% | 63.1% | 31.5% |
| A-537 | 0.079 | **94.0%** | **68.5%** | 31.5% |

These rows are the 314 examples that offer two or more tools. Over all 400 scored examples the largest model reaches
95.3% and 66.3%.

### What this test does and does not show

- **One call format.** The corpus contains two ways of writing a call. Our scorer reads one. Of 1,069 held-out
  trajectories read, 99 contain no call, 269 do not fit in 2,048 tokens and 301 use the format the scorer cannot read.
  The results describe the remaining 400.
- **First action only.** We score the first call of each trajectory. Multi-step behaviour, recovery from a failed
  call and knowing when not to act are not measured.
- **Small sample.** With 314 examples, a difference of two or three points in tool name is within noise. The
  thirteen-point rise in exact arguments from the smallest to the largest model is not.
- **Three points are a trend, not a law.** Loss and both accuracies improved at every size with no sign of a
  plateau. We do not know where it bends.
- **Same distribution.** Held-out examples come from the same sources as training. We have not yet run a public
  benchmark (BFCL, SWE-bench Verified) on these models, and the numbers above must not be set beside benchmark numbers
  of production models as if they measured the same thing.
- **Exact match is harsh.** An argument that is correct but phrased differently counts as wrong. The true rate of
  usable calls is at least the figure reported.

## 5. What we take from it

A model with half a billion parameters, trained for the price of a lunch, on public data, with no instruction
tuning and no chat objective, selects the correct tool among several nineteen times in twenty on unseen tasks of the
kind it was trained on. The curve is still rising at the largest size we could afford this week. That is enough to
justify the next steps, and not enough to justify more than that.

## 6. Next

1. More data before more parameters: the largest model has already seen the corpus twice. An additional 100 GB of open
   code-agent trajectories is being added.
2. A scorer that reads both call formats, and scoring by number of tools offered.
3. Multi-step evaluation, and the public benchmarks, on the same checkpoints.
4. The next size, toward one billion parameters, on the larger corpus.
5. The comparison that matters commercially: this model as the action head of our production agents, against the
   general models they use today, on the tasks our clients actually run.

## References

- Vaswani et al., "Attention Is All You Need", 2017.
- Sutton, "The Bitter Lesson", 2019.
- Karpathy, nanochat (open-source training codebase).
- Toucan-1.5M, tool-use trajectories from real MCP environments (arXiv 2510.01179).
- SWE-smith, trajectories for software-engineering agents.
- Berkeley Function-Calling Leaderboard (BFCL).
- Large Action Models (arXiv 2412.10047).

## Appendix: what is released

- Weights for the three models, the tokenizer, the scorer (`regua_b.py`), the data-preparation and training scripts, and the raw
  training and scoring logs.
- The models are next-token predictors over agent trajectories. They are research artifacts: they are not instruction-tuned,
  have a 2,048-token context, and have not been evaluated for safety or for use in production.
