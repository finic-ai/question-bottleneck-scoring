# AGENTS.md

This repo is a small reference implementation of Question-Bottleneck Scoring (QBS). It is meant to be run as is or adapted to a new decision. It follows the whitepaper "Question-Bottleneck Scoring" (Finic, 2026).

## The approach

Rules and gradient-boosted trees are good at numbers and categories but miss the judgment calls an analyst makes at a glance: whether a name, email and address belong to one real person, whether an address is written the way people there write addresses. LLMs can make those calls but are slow, costly and return text, not probabilities. QBS keeps the judgment and drops the cost:

1. **Calibration.** Collect labeled records in the format available at decision time, with an `is_bad` label.
2. **Question generation.** Write candidate yes/no questions an experienced analyst would ask about a record.
3. **Weighting.** A decision model answers every candidate for every record with a calibrated probability. Forward selection on cross-validated AUC keeps the few questions that predict the label best, and an L2-regularized logistic regression on their standardized log-odds learns their weights.
4. **Scoring.** For a new record, the decision model answers only the kept questions and the frozen weights turn the answers into a score.

In the whitepaper's main study, six questions fitted on 2,000 labeled accounts beat a 300-feature production GBDT where the GBDT was uncertain, and combining the two beat either alone. The questions and their weights are the whole model, so every score is explainable.

## Files

- `qbs/models.py`: decision models. `Grayson` calls Finic's `POST /v1/decide`; `Other` is the stub to implement.
- `qbs/config.py`: `qbs setup` and config loading (`~/.config/qbs/config.json`, or `QBS_CONFIG`).
- `qbs/core.py`: reading records and questions, answering with a cache, forward selection, fitting and scoring.
- `qbs/cli.py`: the `qbs` command.
- `example/`: synthetic sign-ups (`signups.jsonl` labeled, `new_signups.jsonl` not) and candidate questions.

## Running

```bash
pip install -e .
export FINIC_API_KEY=gsk_...   # skips the interactive setup; get one at https://portal.finic.ai/keys
qbs fit example/signups.jsonl example/questions.txt -o model.json
qbs score model.json example/new_signups.jsonl
```

`qbs fit` prints the kept questions with their weights and the cross-validated AUC of the kept set and of all candidates. `model.json` holds the questions, standardization, weights and intercept.

## Adapting it

- **Records:** any JSON object per line, or CSV rows. Include only what was known when the decision was made; anything later leaks the outcome. Every field except the label is sent to the decision model.
- **Label:** `is_bad` by default (`--label` to change). Values `1`, `true`, `yes` or `bad` count as bad. "Bad" is just the positive class: it can be "became a power user".
- **Questions:** one per line. Neutral wording about the record, one thing per question. Holistic questions ("Would a careful person believe these details belong to one real individual?") tend to beat narrow checks that restate a field.
- **Combining with an existing model:** append its score to `Z` in `core.fit` and `core.score`, or use the answers as features in that model.

## Using another decision model

`models.load` returns `Other` when setup chose Other. Implement:

```python
def answer(self, record: dict, questions: list[str]) -> list[float]
```

Return the probability of yes for each question, in order. Probabilities must be calibrated, not verbalized guesses: with a hosted LLM, ask for a one-word yes/no answer and normalize the yes and no token log-probabilities. Read API keys from environment variables, and set `name` so the answer cache doesn't mix models.
