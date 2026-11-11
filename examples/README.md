# Manifest templates

`manifest_template.csv` and `manifest_template.json` hold the same three rows. Copy either one and replace the paths with your own.

| Column | Required | Meaning |
|---|---|---|
| `prediction` | yes | Path to the predicted protein image |
| `ground_truth` | yes | Path to the real protein image |
| `prediction_id` | no | Any text, unique per row. If the column is absent, rows are numbered 1, 2, 3... |
| `mask` | no | Path to a binary cell mask (0 outside the cell, 255 inside). Leave empty (CSV) or `null` (JSON) for no mask; masked metrics are then not reported for that row |
| `cell_id` | no | Groups the predictions of one cell (the challenge asks for several per cell), for per-cell results and the SubCell energy score. Rows of one cell must share `ground_truth`, `mask`, `group_id` and the reference channels. If the column is absent, each row is its own cell |
| `group_id` | no | Groups cells of one protein, usually the antibody (for example `HPA073913`). Needed for brightness spread and for per-protein results (`PREFIX_groups.csv`) |
| `nucleus`, `microtubules`, `er` | no | Paths to the three reference channel images of the cell. All three or none; needed only for SubCell location divergence |

- Relative paths are read from the manifest's own folder. Absolute paths and `~` (your home folder) also work.
- Other columns are ignored, so an existing manifest with extra columns can be reused.
- The format is chosen by the file extension: `.csv` or `.json`.

Run with:

```
cm4ai-evaluate manifest_template.csv -o results/run1
```
