# Self-Supervised Monocular Depth Estimation - Reproduction

A from-understanding reproduction of self-supervised monocular depth estimation on KITTI. I'm building this to learn the geometry that underlies visual depth and SLAM - pixel backprojection, camera pose, and the photometric reprojection loss.

**Status:** In progress — active learning project. Starte 08.04.2026; 

## Papers this is based on

- R. Garg, V. Kumar B G, G. Carneiro, I. Reid — *Unsupervised CNN for Single View Depth Estimation: Geometry to the Rescue* (ECCV 2016)
- C. Godard, O. Mac Aodha, G. Brostow — *Unsupervised Monocular Depth Estimation with Left-Right Consistency* (CVPR 2017)
- C. Godard, O. Mac Aodha, M. Firman, G. Brostow — (Monodepth2, ICCV 2019)

## Running it

So far:

```bash
scripts/download_kitty.sh
uv sync 
uv run warp.py
```

(This section expands as training lands.)

## Results

Eigen-split metrics, filled in as training runs complete. Baseline row to be copied from the Monodepth2 paper's results table for a fair comparison.

| Model              | abs_rel | sq_rel | rmse | δ < 1.25 |
| ------------------ | ------- | ------ | ---- | --------- |
| Monodepth2 (paper) | —      | —     | —   | —        |
| This repo          | —      | —     | —   | —        |

## Notes / what I learned

## License & attribution

Baseline code adapted from `nianticlabs/monodepth2`, © Niantic Inc. 2019, released for non-commercial research use — their license header and attribution are retained wherever their code is used. The KITTI dataset is used under its CC BY-NC-SA terms. If you build on this, please cite the papers listed above.
