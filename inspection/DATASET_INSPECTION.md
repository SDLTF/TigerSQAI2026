# Uploaded archive inspection

- 424 files in total
- 140 RGB images, 140 coarse masks, 140 fine masks
- 10 cases, exactly 14 station frames per case
- 126 images at 1920×1080 and 14 images at 1280×720
- Coarse masks: all 16 classes occur
- Fine masks: 31 declared classes, but Right bronchial artery, Gastric conduit, and Omentum have zero annotated pixels
- Every RGB mask color matched `labelmap.csv`; no unknown mask colors were found

The flattened archive mapping is:

```text
name.png     -> coarse mask
name(1).png  -> fine mask
name(2).png  -> RGB image
```

See the two CSV files in this folder for exact pixel counts and image occurrence rates.
