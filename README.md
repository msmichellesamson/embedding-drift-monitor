# embedding-drift-monitor

> Detecting when an embedding model's output distribution has quietly shifted underneath you.

## The question I'm exploring

When you fine-tune an embedding model, swap to a new version, or just let one
run for a long time against a non-stationary input distribution, the geometry
of the embedding space changes. Downstream retrieval breaks in subtle ways:
recall@k drops, similarity thresholds you tuned six months ago start
under- or over-firing, and your evals don't notice because they're computed
on a fixed test set.

Anthropic's *Mapping the Mind of a Large Language Model* showed how
**internal feature representations have real, measurable structure** — and
implicitly, that structure can shift. I started wondering: how would you
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

- Statistical tests on raw embedding dimensions are noisy. KS on individual
  dimensions flags drift constantly because most embedding dimensions aren't
  individually meaningful. MMD on whole vectors is more stable but expensive.
- The right unit of comparison probably isn't "is the distribution different?"
  but "**did downstream retrieval quality change?**" — which means I need
  ground-truth retrieval pairs, not just unlabelled embeddings.
- Per-cluster drift is more informative than global drift. If a specific
  semantic region of the embedding space shifts, that's actionable; if the
  whole space drifts uniformly, it's probably a model swap.
- The alerting code is more mature than the detection code, which is a fair
  reflection of where I spent time. The hard part of monitoring isn't
  finding anomalies — it's deciding which ones to wake someone up for.

## What I'd do next

- Build a retrieval-quality canary: a fixed set of (query, expected-doc)
  pairs that I run continuously and treat the recall@k as the primary
  signal. Distribution drift becomes a secondary explanatory metric.
- Test on a real production embedding service. Synthetic drift is too easy
  to detect.
- Add a "drift root cause" view that correlates drift with model version
  changes, traffic mix changes, and time-of-day patterns
- Look at SAE feature activations as a more interpretable drift signal, in
  the spirit of Anthropic's interpretability work

## Status

The pipeline runs. The detection thresholds are guesses. The alerting works
and is the part I'm most happy with.

## References

- Templeton et al., *Scaling Monosemanticity: Extracting Interpretable Features
  from Claude 3 Sonnet* (Anthropic, 2024)
- Gretton et al., *A Kernel Two-Sample Test* (JMLR, 2012)
- Rabanser et al., *Failing Loudly: An Empirical Study of Methods for
  Detecting Dataset Shift* (NeurIPS 2019)
