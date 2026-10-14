
# CM4AI-AnVIL IF image generation evaluator



```
python ./evaluate.py ./results/predictions ../test_data -o stats.json
```


Notes: 
 - Because of upstream limitations, current version looks for .tiff predictions and .png truth files.
## Open issues

- **Cosine similarity rewards bright, featureless predictions.** `cosine_similarity` is not mean centred, so a uniformly bright image looks fairly similar to most cell images. On the 200 sample images (2026-10-13), an all-white prediction (every pixel 255) scored 0.675 against ground truth, higher than 199 of 200 wrong-protein pairs (median 0.30) and close to a real BLAST-ProtiCelli prediction (0.708). Mean absolute error does flag it (219, against a median of 21.5 for wrong-protein pairs). Relevant if cosine carries weight in the challenge score.
- **Mutual information is less reliable for small masks.** It is estimated from a 256 x 256 histogram of paired intensities. With few pixels inside the mask, most histogram cells are empty or hold one pixel, and the estimate is biased upward. scikit-learn's warning about this ("number of unique classes is greater than 50% of the number of samples") is silenced in the tool because it describes classification labels, not intensity bins; this note replaces it.
- **Very small masks.** When the mask's bounding box is smaller than 11 x 11 pixels, masked MS-SSIM is reported as null with a note. Masks this small may be debris or dead cells rather than whole cells (unverified). Growing the mask is out of scope.
