# Metrics

Every metric compares a predicted protein image with the real (ground truth) image. Each is reported twice: on the whole image (`unmasked_`) and inside the cell mask (`masked_`), when a mask is given.

None of these metrics is final. The evaluation team is still deciding which metrics the challenge will use and how to combine them.

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
- **The other metrics** are ported from the ProtiCelli quality evaluation notebook, [`analysis/quality_evaluation/evaluate_predictions.ipynb`](https://github.com/CellProfiling/ProtiCelli/blob/1e876ef/analysis/quality_evaluation/evaluate_predictions.ipynb) (commit 1e876ef), on images scaled to [0, 1]. They give identical values to the notebook's own functions on 48 real image pairs. Deliberate changes: constant images are scored rather than skipped; Gabor responses are computed with Fourier transforms (5 times faster, same values to rounding); Inception runs in batches and can use Apple GPUs.

## Packages each metric needs

| Install | Metrics |
|---|---|
| `pip install cm4ai-evaluate` | the nine statistics, Pearson, Spearman, PSNR |
| add `[extended]` | MS-SSIM, mutual information, Haralick, Gabor |
| add `[inception]` | Inception L2 (also needs `--inception` when running) |

Metrics whose packages are missing are left out, and the run summary says which and how to install them.
