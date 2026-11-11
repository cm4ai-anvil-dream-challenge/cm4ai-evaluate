# cm4ai-evaluate

Score predicted protein images against ground truth images for the CM4AI DREAM Challenge. A command-line tool for participants and organizers to run locally, for example on a locally trained model. Not the challenge's official grading, which runs on the organizers' side.

**Status:** draft for review. None of the metrics is final; the evaluation team is still deciding which metrics the challenge will use and how to combine them. See [open issues](docs/open-issues.md).

## Install

Python 3.10 or later. From this folder, ideally in a new environment:

```
pip install -e ".[extended]"
```

| Install | Metrics |
|---|---|
| `pip install -e .` | the nine grading statistics, Pearson, Spearman, PSNR |
| `pip install -e ".[extended]"` | adds MS-SSIM, mutual information, Haralick, Gabor |
| `pip install -e ".[extended,subcell]"` | adds the SubCell panel metrics (needs torch; downloads 349 MB of model weights on first use; a GPU is much faster) |
| `pip install -e ".[extended,inception]"` | adds Inception L2 (needs torch; downloads about 100 MB of model weights on first use) |

## Quick start

1. Make a manifest listing each prediction with its ground truth and, optionally, a cell mask, cell and protein identifiers, and reference channels. See the templates and column guide in [examples/](examples/).
2. Run:

   ```
   cm4ai-evaluate manifest.csv -o results/run1
   ```

| Option | Meaning |
|---|---|
| `--mode strict` (default) | Inputs must meet the challenge image specification: PNG, 512 x 512, 8-bit grayscale |
| `--mode lenient` | Accepts common departures (TIFF, 16-bit, other sizes and more) and records each one. See [input checks](docs/input-checks.md) |
| `-o PREFIX` | Writes `PREFIX.jsonl`, `PREFIX.csv`, `PREFIX_cells.csv` and, with `group_id`, `PREFIX_groups.csv` (default `results`) |
| `--metrics LIST` | Tiers and metric names to run, for example `basic` (fast) or `basic,subcell`. Default `basic,extended`. See [choosing metrics](docs/metrics.md#choosing-metrics) |
| `--min-cells N` | Flags proteins with fewer than N complete cells in `PREFIX_groups.csv` |
| `--subcell-weights FILE` | Use an existing SubCell weights file instead of downloading |
| `--location-head FILE` | Location head for SubCell location divergence (not public at present) |
| `--inception` | Also computes Inception L2 |
| `--device cpu\|cuda\|mps` | Device for SubCell and Inception (default: whichever is available) |

A row that fails a check is recorded with the reason and the run continues. Exit code 0 means every row was scored, 1 that some rows were not, 2 that the manifest or setup could not be used.

## Outputs

- **`PREFIX.csv`**, for people: one row per prediction, with every pair metric twice (`unmasked_` and `masked_` columns), panel metrics once (`masked_`), deviations, notes and errors.
- **`PREFIX_cells.csv`**: one row per cell, with per-image metrics averaged over the cell's predictions and the per-cell metrics.
- **`PREFIX_groups.csv`** (with `group_id`): one row per protein, averaged over its complete cells. Per-protein averages need enough cells to mean anything; see [three levels of assessment](docs/metrics.md#three-levels-of-assessment).
- **`PREFIX.jsonl`**, for further analysis: one JSON record per prediction, shaped like the challenge grading rows (`prediction_id` plus a `metrics` object), with `masked_metrics`, `deviations`, `notes` and `error` alongside.

## Metrics

| Group | Metrics |
|---|---|
| Challenge grading statistics | absolute, squared and maximum error, mean absolute error, RMSE, cosine similarity, and the sums they come from |
| Pixel correlation | Pearson, Spearman |
| Image quality | PSNR, MS-SSIM |
| Information and texture | mutual information, Haralick, Gabor |
| Learned features | Inception L2 (optional) |
| Preliminary challenge panel | brightness spread; SubCell pattern distance, location divergence, energy score (optional) |

Each value is reported raw: the tool does not weight or combine metrics. Which direction is better for each, what each measures, how masks are applied, and where each comes from: [docs/metrics.md](docs/metrics.md). No single metric is enough; for example, dim images score well on error-based metrics whatever their pattern.

## Verification

- The grading statistics reimplement the challenge grader in `cm4ai-anvil-dream-tools` (cited by commit in the code) and give identical values on 203 image pairs.
- The other metrics are ported from the ProtiCelli quality evaluation notebook and give identical values to its own functions on real image pairs.
- The panel metrics reimplement the evaluation team's preliminary panel (`dream_eval` 0.3.0) and give identical values to its functions: brightness spread on 400 images; SubCell features, bit for bit, on 32 images.
- Run on all 200 sample cells with BLAST-ProtiCelli predictions, masked and unmasked, with Inception.

## Development

```
pip install -e ".[extended,inception,dev]"
pytest
```

## Documentation

- [docs/metrics.md](docs/metrics.md): metric primer
- [docs/input-checks.md](docs/input-checks.md): strict and lenient checks, masks
- [docs/open-issues.md](docs/open-issues.md): questions for the evaluation team
- [examples/](examples/): manifest templates
