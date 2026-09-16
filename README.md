# GreenTrust-Crisis

Code and results for **"GreenTrust-Crisis: Ontology-Grounded, Group-Conditional
Conformal Prediction for Trustworthy Multilingual Crisis Triage"**, accepted at
AI4S 2026 (Springer).

## What the paper shows

A split-conformal layer on a distilled multilingual crisis classifier reports
0.909 coverage overall and looks safe. It is not. On the same test set it covers
the critical `infrastructure` class only 0.526 of the time (95% CI [0.29, 0.76]),
and French and Haitian Creole below 0.89. The marginal guarantee holds on average
over a stream that you should not be averaging over.

Conditioning the calibration on groups fixes it. Per-class calibration lifts
`infrastructure` to 0.895 (McNemar p = 0.016) and per-language calibration lifts
French to 0.935 (p < 0.001) at no cost in set size. The groups come from a
taxonomy derived from the empathi, MOAC and HXL crisis vocabularies rather than
from anything we invented.

The paper also introduces a **concept-smoothed calibrator**, which shrinks each
class threshold towards the pooled threshold of its ontology concept with weight
`lambda_k = n_k / (n_k + tau)`. `tau` is chosen by cross-validation inside the
calibration set, so the test set is never touched. It reproduces class-conditional
coverage at a smaller mean set size, 2.90 against 3.03.

Across 14 model runs over four backbones, marginal coverage of the rare class
stayed at or below 0.526 and class-conditional coverage at or above 0.842, with no
overlap.

## Layout

```
src/            the pipeline: data, features, distillation, conformal, energy
scripts/        the camera-ready experiments, one script per result
beam/           the GPU jobs for the mBERT and XLM-R baselines
results/        every number in the paper, as JSON, plus cached probabilities
configs/        default hyperparameters
```

## Reproducing

```bash
pip install -r requirements.txt
python scripts/download_data.py        # fetches the corpus from Hugging Face
```

Most results do not need the corpus. `results/cache.npz` holds the calibration
and test probabilities of the deployed student, and `results/transformer_probs.npz`
holds the transformer probabilities, so the conformal analysis and every figure
run directly:

```bash
python scripts/cr_conformal.py         # Tables 3 to 5, all conformal variants
python scripts/cr_tau_cv.py            # tau by cross-validation on calibration only
python scripts/cr_significance.py      # McNemar and paired bootstrap
python scripts/cr_robustness.py        # the tau sweep behind Fig. 4
python scripts/cr_fig_camera.py        # all 7 figures, vector PDF, into figures/
```

These need the corpus, because they refit the models:

```bash
python scripts/cr_base.py              # teacher, hard student, distilled student
python scripts/cr_temp.py              # the distillation temperature sweep
python scripts/cr_latency_linear.py    # same-host inference latency and energy
```

The transformer baselines were trained on a single RTX 4090 through Beam. The job
definitions are in `beam/`. Everything else runs on one CPU core.

## Dataset

The Figure Eight (now Appen) multilingual disaster response corpus, 26,248
messages, from
[Hugging Face](https://huggingface.co/datasets/community-datasets/disaster_response_messages).
It is not redistributed here; `scripts/download_data.py` fetches it.

One detail worth knowing if you re-run the split: the three splits sum to 26,247,
not 26,248. The splitter drops any (language, class) stratum with fewer than two
members, and exactly one message qualifies, a French `infrastructure` message that
is the only one of its kind.

## Energy numbers

Energy is modelled, not measured. RAPL counters were not exposed on either host,
so energy is `65 W x measured latency`. Only latency is measured, which is why the
paper quotes energy to two significant figures and the cross-model gap as an order
of magnitude rather than a precise ratio. Table 6 in the paper reports only models
timed on the same host, so no row in it is compared across machines.

## Citation

```bibtex
@inproceedings{pettugani2026greentrust,
  title     = {GreenTrust-Crisis: Ontology-Grounded, Group-Conditional Conformal
               Prediction for Trustworthy Multilingual Crisis Triage},
  author    = {Pettugani, Hotragn and Pettugani, Tirdesh},
  booktitle = {Artificial Intelligence: Towards Sustainable Intelligence},
  series    = {Communications in Computer and Information Science},
  publisher = {Springer},
  year      = {2026},
  note      = {Accepted, to appear}
}
```

## Licence

MIT, see [LICENSE](LICENSE). The corpus carries its own terms; see the dataset
page linked above.
