# Input checks

Each prediction and ground truth image is checked before scoring. Prediction and ground truth are checked the same way.

- **strict** (default): the challenge's image specification. A file that fails is not scored; the row is recorded with `status: error` and the reason, and the run continues.
- **lenient** (`--mode lenient`): accepts common departures from the specification where the meaning is unambiguous. Each one is recorded as a deviation in the output, and the row gets `spec_compliant: false`. Deviations marked as warnings are also listed in the run summary.

| Check | strict | lenient |
|---|---|---|
| File exists, decodes, no NaN or infinity | error | error |
| Not a symbolic link (a security rule from the challenge grading) | error | allowed (`symlink`) |
| PNG, judged by content rather than file name | error | TIFF allowed (`tiff`); JPEG allowed with a warning that it is lossy (`jpeg`); other formats error |
| Single channel (PIL mode `L`) | error | extra length-1 axes removed (`singleton_axes`); RGB, RGBA or grayscale plus alpha with identical colour planes reduced to one, with a warning (`collapsed_channels`); palette PNGs expanded (`palette`). Error if planes differ, alpha is not fully opaque, or the image is 1-bit |
| Prediction and ground truth the same size | error | error |
| 512 x 512 | error | other sizes allowed, with a warning (`size`) |
| 8-bit (uint8, 0 to 255) | error | 16-bit and other unsigned integers divided by their maximum, with a warning (`bit_depth`; plus `fits_12_bit` if a 16-bit image never exceeds 4,095, since it will appear dim); floats accepted only within [0, 1]; signed integers and booleans error |

## Masks

Masks are optional. A mask that fails a check never stops the run: the masked metrics for that row are skipped and `mask_skip_reason` says why.

| Check | strict | lenient |
|---|---|---|
| Same file checks as the images | as above | as above |
| Same size as the images | required | required |
| Values | exactly 0 (outside the cell) and 255 (inside) | also 0 and 1, 0 and the type maximum, 1-bit, or boolean (`mask_values`, `one_bit`) |
| At least one pixel inside the cell | required | required |

Contributions are welcome for other cases that can be handled without guessing.
