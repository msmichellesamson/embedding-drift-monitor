# embedding-drift-monitor

> Detecting when an embedding model's output distribution has quietly shifted underneath you.

## The question I'm exploring

When you fine-tune an embedding model, swap to a new version, or just let one
run for a long time against a non-stationary input distribution, the geometry
of the embedding space changes. Downstream retrieval breaks in subtle ways:
recall@k drops, similarity thresholds you tuned six months ago start
under- or over-firing, and your evals don't notice because they're computed
on a fixed test set.

Anthropic's [*Mapping the Mind of a Large Language Model*](https://www.anthropic.com/research/mapping-mind-language-model)
showed how **internal feature representations have real, measurable
structure** — and implicitly, that structure can shift. I started wondering: how would you
build a monitor that catches that shift early, before it becomes a quality
incident?

## Why I care

This is one of the silent-failure modes I worry about most in production AI
systems. There's no exception, no alert, no failed test — just slowly
degrading retrieval quality that nobody attributes to embedding drift until
weeks later. It's the kind of issue good observability should catch.

I also wanted to build something that took **alerting** seriously. Most
monitoring projects stop at "emit a Prometheus metric." Real on-call
requires routing to the right channel with the right severity and not
spamming people. That part is genuinely hard and worth the effort.

## What's in here

The detection pipeline:

- `src/analysis/statistical_tests.py` — KS, MMD, and Wasserstein-based tests
  over embedding distributions
- `src/analysis/anomaly_detector.py` — per-dimension distributional checks
- `src/analysis/time_series_detector.py` — rolling-window change-point detection
- `src/core/drift_detector.py` — orchestration + thresholding

The alerting surface:

- `src/alerts/` — Slack, PagerDuty, Teams, Email, Discord, generic webhook
  notifiers, all with retry + circuit-breaker wrappers and a severity manager
  that does deduplication and escalation

Plus the standard infra: PostgreSQL for metadata, Redis for hot lookups,
k8s with RBAC and HPA, Prometheus + Grafana, structured logging.

## What I'm finding (so far)

I ran a controlled comparison on [AG News](https://huggingface.co/datasets/ag_news)
with three drift detectors and four test distributions ranging from no
drift to total topic swap. Full writeup in [`experiments/findings.md`](experiments/findings.md).

**Detection (α = 0.01):**

| Detector | control | mild (20% drift) | moderate (50%) | severe (100%) |
|---|---|---|---|---|
| per-dim KS (Bonferroni) | no drift | **MISS** | FLAG | FLAG |
| MMD with permutation | no drift | FLAG | FLAG | FLAG |
| classifier two-sample | no drift | FLAG | FLAG | FLAG |

**Wall-clock per check:** classifier two-sample 0.01s · per-dim KS 0.07s · MMD 0.35s.

- **Per-dim KS missed mild drift entirely.** Even with 20% of the test
  distribution swapped to a different topic, no individual dimension
  cleared the Bonferroni-corrected threshold. This is the case the
  monitor most needs to catch and the one this method handles worst.
- **The [classifier two-sample test](https://arxiv.org/abs/1610.06545)
  is both the most sensitive AND the fastest** — about 30x faster than
  MMD and the only one besides MMD that catches mild drift. That was
  not what I expected before running this.
- **MMD's score is the most useful severity signal.** MMD² grows
  monotonically from 0.0002 (control) to 0.066 (severe), giving a
  dimensionless number you can compare across runs. Classifier
  accuracy is bounded by task separability, so its absolute value
  is harder to read across deployments.
- **No false positives on the control distribution** for any of the
  three. That was the table-stakes check.

## What I'd do next

- **Replace per-dim KS with the classifier two-sample test as the
  primary detector** in `src/analysis/statistical_tests.py`, and keep
  MMD as the secondary "is this serious?" signal. Per-dim KS gives
  false confidence on the case that matters most (mild drift).
- Add a **retrieval-quality canary**: a fixed set of (query, expected-doc)
  pairs scored continuously. Distributional drift becomes the
  explanatory metric for canary regressions, not the primary alarm.
- Repeat the experiment with **per-cluster drift** — drift that only
  affects one semantic region of the embedding space. AG News topic
  swap is too clean a case; sub-topic drift inside a single category
  is the harder, more realistic test.
- Look at [SAE feature activations](https://transformer-circuits.pub/2024/scaling-monosemanticity/)
  as a more interpretable drift signal — whether feature-level drift
  is more diagnostic than raw embedding-level drift is genuinely an
  open question to me.

## Status

The pipeline runs end-to-end. The detection comparison ([`experiments/findings.md`](experiments/findings.md))
is real and reproducible. The alerting code is more mature than the
detection code, which is a fair reflection of where I spent time —
the hard part of monitoring isn't finding anomalies, it's deciding
which ones to wake someone up for.

## References

- Templeton et al., [*Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet*](https://transformer-circuits.pub/2024/scaling-monosemanticity/) (Anthropic, 2024)
- Anthropic, [*Mapping the Mind of a Large Language Model*](https://www.anthropic.com/research/mapping-mind-language-model) (2024)
- Gretton et al., [*A Kernel Two-Sample Test*](https://jmlr.org/papers/v13/gretton12a.html) (JMLR, 2012)
- Rabanser et al., [*Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift*](https://arxiv.org/abs/1810.11953) (NeurIPS 2019)
- [Alibi Detect](https://github.com/SeldonIO/alibi-detect) — drift detection library used as a reference implementation
