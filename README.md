# Question-Bottleneck Scoring (QBS)

A reference implementation of QBS, described in the whitepaper [Question-Bottleneck Scoring](WHITEPAPER_URL).

QBS turns a decision into plain-language yes/no questions. A decision model answers each question with a calibrated probability, and a logistic regression weights the answers into a score. Every score traces back to a few answers a person can check.

## Install

```bash
pip install -e .
qbs setup
```

Setup asks which decision model answers the questions. The default is [Finic Grayson](https://finic.ai): paste an API key from [portal.finic.ai/keys](https://portal.finic.ai/keys) and you're done. Choose Other to use your own model (see [Using another model](#using-another-model)).

## Quickstart

```bash
qbs fit example/signups.jsonl example/questions.txt -o model.json
qbs score model.json example/new_signups.jsonl
```

`fit` answers every candidate question for every labeled record, keeps the questions that best predict the label, and fits their weights. `score` answers only the kept questions for new records and prints one JSON line per record:

```json
{"score": 0.15, "answers": {"Does the account look like an ordinary consumer signing up for themselves?": 0.85, "...": 0.92}}
```

## The four steps

1. **Calibration.** Prepare labeled records in the format you'll score: one record per line (`.jsonl`) or row (`.csv`), as known at the moment of the decision, with an `is_bad` field. Aim for at least 100 records with 20 of each outcome.
2. **Question generation.** Write candidate yes/no questions, one per line. Ask about the record, not what to decide.
3. **Weighting.** `qbs fit` converts the answers to log-odds, standardizes them and uses forward selection on 5-fold cross-validated AUC to choose questions for an L2-regularized logistic regression.
4. **Scoring.** `qbs score` answers the chosen questions and applies the frozen weights.

Answers are cached next to the records (`*.answers.jsonl`), so refitting with new or edited questions only pays for what changed.

## Using another model

Choose Other in `qbs setup`, then make `Other.answer` in [`qbs/models.py`](qbs/models.py) return the probability of yes for each question:

```python
def answer(self, record: dict, questions: list[str]) -> list[float]: ...
```

## License

MIT
