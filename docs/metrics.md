# Metrics

Every metric compares predicted protein images with the real (ground truth) images. There are two kinds:

- **Pair metrics** compare one prediction with its ground truth. Each is reported twice: on the whole image (`unmasked_`) and inside the cell mask (`masked_`), when a mask is given.
- **Panel metrics** are the measures in the evaluation team's preliminary metric panel. They are computed inside the cell mask only, and some need more than one image pair: the other cells of the same protein, or several predictions of the same cell.

None of these metrics is final. The evaluation team is still deciding which metrics the challenge will use and how to combine them. This tool reports each metric's raw value and does not combine them (see [Combining metrics](#combining-metrics)).

## Three levels of assessment

| Level | Unit | Output file | Metrics |
|---|---|---|---|
| Per image | one prediction against its ground truth | `PREFIX.csv`, `PREFIX.jsonl` | every pair metric; brightness spread, SubCell pattern distance, SubCell location divergence |
| Per cell | all predictions of one cell (the challenge asks for several per cell) | `PREFIX_cells.csv` | SubCell energy score; also every per-image metric averaged over the cell's predictions |
| Per protein | all cells of one protein (`group_id`, usually the antibody) | `PREFIX_groups.csv` | every per-cell value averaged over the protein's complete cells |

A cell is **complete** when all its predictions were scored. As in the challenge grading, a cell's average of a metric is empty when any of its predictions lacks that value, and a protein's averages use complete cells only. `PREFIX_groups.csv` also gives, for each panel metric, how many cells had a value (`<metric>_n_cells`).

Brightness spread is a per-image value but needs context: it compares a prediction with the other cells of the same protein.

### How many cells a protein needs

A per-protein average reflects a model's performance on that protein only when it covers enough cells. **The minimum is open** (to be confirmed with the evaluation team, with its source). With `--min-cells N`, `PREFIX_groups.csv` gains an `enough_cells` column marking proteins with at least N complete cells. Treat averages over fewer cells as anecdotes, not assessments. The training tutorial's quick mode, with a handful of held-out cells per antibody, is a check that the evaluation runs, not an assessment.

## Which direction is better

"Better" means closer to the ground truth.

| Metric | Better | Range | What it measures |
|---|---|---|---|
| `absolute_error_sum`, `squared_error_sum`, `dot_product_sum`, `prediction_square_sum`, `ground_truth_square_sum` | neither | 0 and up | Building blocks for the metrics below. They grow with the number of pixels, so masked and unmasked values cannot be compared directly |
| `maximum_absolute_error` | lower | 0 to 255 | The single worst pixel. Noisy: one hot pixel decides it |
| `mean_absolute_error` | lower | 0 to 255 | Average intensity error per pixel |
| `root_mean_squared_error` | lower | 0 to 255 | Like mean absolute error, but large errors count more |
| `cosine_similarity` | higher | 0 to 1 for these images | Overlap of the two images as vectors. Not mean centred, so overall brightness raises it (see [open issues](open-issues.md)) |
| `pearson_r` | higher | -1 to 1 | Whether the spatial pattern matches, ignoring overall brightness and offset |
| `spearman_r` | higher | -1 to 1 | As Pearson, but on pixel ranks, so it tolerates nonlinear intensity differences |
| `psnr_db` | higher | decibels, often 15 to 45 | Peak signal-to-noise ratio: mean squared error on a log scale relative to the maximum value. Infinite for identical images, written as null with a note |
| `ms_ssim` | higher | 0 to 1 | Multi-scale structural similarity: local brightness, contrast and structure, compared at five scales |
| `mutual_information_nats` | higher | 0 up to the image's own entropy | How much knowing one image's intensities tells you about the other's, for any relationship, not only linear. Its ceiling differs between images (3.9 for one identical pair in the sample data), so compare it within an image rather than across images |
| `haralick_similarity` | higher | up to 1 | Similarity of six texture statistics from the grey level co-occurrence matrix (contrast, dissimilarity, homogeneity, energy, correlation, angular second moment). Insensitive to where the texture is |
| `gabor_similarity` | higher | -1 to 1 | Correlation of texture maps from Gabor filters at 3 frequencies and 4 orientations. Sensitive to location |
| `inception_l2` | lower | 0 and up, arbitrary scale | Distance between features from an InceptionV3 network trained on everyday photographs (ImageNet): a rough perceptual similarity. The network has not seen microscopy. Needs `--inception` |

## Panel metrics

From the evaluation team's preliminary panel `C3_hist` (see [Where each metric comes from](#where-each-metric-comes-from)). Every value here is oriented **lower is better**. Names are descriptive; the panel's own names are listed for reference.

| Metric | Category | Level | Range | What it measures | Needs | Panel name |
|---|---|---|---|---|---|---|
| `brightness_spread_w1` | Brightness spread | image | 0 to 1 | Whether the spread of intensities inside the cell matches the protein's real cells. Each image is divided by its own in-mask 99th percentile, then its intensity histogram is compared (Wasserstein-1 distance) with the mean histogram of the protein's *other* cells. Measures the shape of the intensity distribution, not overall brightness | `group_id`; at least one other cell of the protein | `hist_w1_si` |
| `subcell_pattern_distance` | Pattern similarity | image | 0 to 2 | Cosine distance between SubCell features of the prediction and of the ground truth, protein channel only (scaled to its own minimum and maximum, so overall brightness drops out) | `[subcell]` | `emb_dist_subcell` |
| `subcell_location_divergence` | Location | image | 0 to ln 2 (0.693) | Whether a location classifier assigns the prediction to the same compartments as the ground truth. SubCell features of all four channels (protein scaled with the reference channels, so its brightness relative to them counts) go through a linear multi-label head; the two probability sets are compared by Jensen-Shannon divergence | `[subcell]`; `nucleus`, `microtubules`, `er` columns; `--location-head` | `loc2_kolfhead_jsd`, which is minus this value |
| `subcell_energy_score` | Variability/accuracy | cell | 0 and up | Fair energy score of a cell's predictions in SubCell feature space: their mean distance to the truth, minus half their mean distance to each other. Rewards predictions that are close to the truth and whose spread reflects real uncertainty; identical predictions get no credit for spread. With one prediction per cell it measures distance to the truth only | `[subcell]`; several predictions per cell (`cell_id`) for its full meaning | `q3m_es_fair_ludvig_subcell` |

Undefined values are empty, with a note: brightness spread when a prediction has no signal inside the mask (99th percentile at or below 1/1020) or the protein has no other usable cell; SubCell metrics when the mask is empty. In the challenge grading a missing value counts as the worst score, so an empty value here is not neutral.

**The location head is not public at present.** It is part of the evaluation team's private scoring files, and its classes are not documented. Without `--location-head`, `subcell_location_divergence` is skipped.

**SubCell** ([Gupta et al. 2024](https://doi.org/10.1101/2024.12.06.627299)) is a vision transformer trained on Human Protein Atlas images. The tool uses the public rybg MAE-CellS-ProtS-Pool encoder, downloaded once (349 MB) and checked against a fixed checksum. Features are computed in 32-bit floats; the challenge grading uses 16-bit floats on CUDA GPUs, which the evaluation team reports changes feature cosine similarity by less than 0.0001.

## Combining metrics

The challenge will combine metrics into one benchmark score to rank models: per metric, a protein's average is put on a common scale, categories are weighted, and proteins are averaged. The scaling constants and weights are not public and not final; the plan is to set them after an initial submission round, across models. This tool therefore reports raw values only. Adding raw values of different metrics is not meaningful: they are in different units.

## Read several metrics together

No single metric is enough. In a run of four BLAST-ProtiCelli predictions (2026-10-13), prediction 199 had the best values on the error-based metrics (mean absolute error 1.6 where the others were 8 to 12; PSNR 41.8 dB where the others were 21 to 25; MS-SSIM 0.82; the lowest Inception distance) and almost no agreement on the pattern metrics (Pearson 0.03, mutual information 0.001, Gabor 0.06).

Its ground truth is very dim, with an average intensity of about 1.4 out of 255. A dim prediction is close in absolute terms whatever its pattern, so error-based metrics reward it; Pearson, which ignores brightness, shows that the localization is not captured. For dim proteins, error metrics and PSNR mostly measure brightness.

## Masked versions

| Method | Metrics | What it uses |
|---|---|---|
| Pixel subset | the nine statistics, Pearson, Spearman, PSNR, mutual information | Only pixels inside the mask. Sums count only those pixels |
| Exact co-occurrence | Haralick | Only neighbouring pixel pairs with both pixels inside the mask. Outside pixels are given a separate grey level whose row and column are removed from the co-occurrence matrix, so a mask covering the whole image gives exactly the unmasked value |
| Bounding box | MS-SSIM, Gabor, Inception | Both images cropped to the smallest rectangle containing the mask. These need a rectangular image, so background inside the rectangle is included |

When a cell touches all four edges of the image, its bounding box is the whole image and the bounding-box metrics equal their unmasked values. When the bounding box is smaller than 11 x 11 pixels, masked MS-SSIM is null with a note.

## Undefined values

Metrics that are undefined for an input are written as null, with the reason in `notes`:

- Pearson, Spearman, Haralick and Gabor are undefined when an image (or the region inside the mask) is constant, every pixel the same value, for example an all-black prediction.
- Cosine similarity is undefined when an image is all zero.
- PSNR is infinite for identical images.

Mutual information with a constant image is a true 0, not undefined, and is reported as 0 with a note.

## Where each metric comes from

- **The nine statistics** reimplement the challenge grading in [cm4ai-anvil-dream-tools](https://github.com/cm4ai-anvil-dream-challenge/cm4ai-anvil-dream-tools): `organizer/grading/comparison.py` (commit 3a6ec06) and `participant/benchmarking/metrics.py` (participant commit f4c5661). They are computed on raw 0 to 255 values, as the grading does, and were checked to give identical values on 203 image pairs.
- **The panel metrics** reimplement the evaluation team's preliminary panel as packaged in `dream_eval` 0.3.0 (cm4ai-anvil-dream-tools, branch `refactor/shared-organizer-manifest-subsets`, commit 581edc9; its math is copied from the evaluation team's spec module `c3metrics.py`, version 1.0 of 2026-10-20). The SubCell model code is copied from the same package. Checked against `dream_eval`'s own functions on the sample cells and BLAST-ProtiCelli predictions (2026-11-10, 32-bit floats on a CPU): brightness spread identical on 400 images (200 predictions and their ground truths); SubCell features identical, bit for bit, for both inputs on 32 images; pattern distance equal to rounding (3e-16); location divergence (with a random test head) and energy score identical.
- **The other metrics** are ported from the ProtiCelli quality evaluation notebook, [`analysis/quality_evaluation/evaluate_predictions.ipynb`](https://github.com/CellProfiling/ProtiCelli/blob/1e876ef/analysis/quality_evaluation/evaluate_predictions.ipynb) (commit 1e876ef), on images scaled to [0, 1]. They give identical values to the notebook's own functions on 48 real image pairs. Deliberate changes: constant images are scored rather than skipped; Gabor responses are computed with Fourier transforms (5 times faster, same values to rounding); Inception runs in batches and can use Apple GPUs.

## Packages each metric needs

| Install | Metrics |
|---|---|
| `pip install cm4ai-evaluate` | the nine statistics, Pearson, Spearman, PSNR |
| add `[extended]` | MS-SSIM, mutual information, Haralick, Gabor |
| add `[subcell]` | SubCell pattern distance, location divergence, energy score (torch; weights downloaded on first use) |
| add `[inception]` | Inception L2 |

## Choosing metrics

`--metrics` takes a comma-separated list of tiers and metric names, for example `--metrics basic` or `--metrics basic,subcell_pattern_distance`.

| Tier | Install | Speed | Metrics |
|---|---|---|---|
| `basic` | none | fast: about 0.05 s per prediction | the nine statistics, Pearson, Spearman, PSNR, brightness spread |
| `extended` | `[extended]` | about 0.5 s per prediction | MS-SSIM, mutual information, Haralick, Gabor |
| `subcell` | `[subcell]` | about 2 s per image on a 2-core CPU; far faster on a GPU | SubCell pattern distance, location divergence, energy score |
| `inception` | `[inception]` | moderate | Inception L2 (also `--inception`) |
| `all` | | | every tier |

The default is `basic,extended`. A metric name selects the entry that computes it; the nine statistics come from shared sums, so naming one computes all nine. With the default, metrics whose packages are missing are left out with a message; when you name metrics or tiers explicitly, missing packages stop the run. Panel metrics whose inputs are absent (no `group_id`, no reference channels, no location head) are skipped with a message.
