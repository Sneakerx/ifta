# IFTA-Overdrive

Reference implementation and stability analysis of the **Overdrive** variant of the iterative Fourier transform algorithm (IFTA) for phase-only hologram synthesis.

This repository accompanies the preprint
*"Stability and convergence of the Overdrive variant of the iterative Fourier transform algorithm"* (Bernau, in preparation).

## What's here

```
ifta-overdrive/
├── ifta/                 Python package with all algorithm variants
│   ├── algorithms.py     GS, Fienup-Amplitude, Bengtsson, Overdrive, ...
│   ├── adaptive.py       Adaptive β selection from local contraction estimate
│   ├── metrics.py        PSNR, MSE, log-spectral / Itakura–Saito divergence
│   └── utils.py          Common helpers (normalization, shifting, masks)
├── scripts/
│   ├── smoke_test.py     End-to-end run on a single image
│   ├── benchmark.py      Compare all algorithms across an image set
│   └── plot_results.py   Render figures used in the paper
├── paper/                LaTeX preprint (article class, arXiv-compatible)
├── tests/                pytest unit tests
└── data/                 Test images (not version-controlled — add your own)
```

## Quickstart

```bash
# 1. Install (Python 3.10+)
pip install -e .

# 2. Place test images in data/ (e.g. the "Klasse 1" set from the dissertation)

# 3. Smoke test
python scripts/smoke_test.py data/lighthouse.bmp

# 4. Full benchmark
python scripts/benchmark.py data/ --iterations 20 --output results/
```

## The Overdrive update rule

For phase-only hologram synthesis with target amplitude $|F|$, current
reconstruction $|G_k|$, and previous virtual amplitude $|G_{k-1}'|$:

$$
|G_k'| = |G_{k-1}'|^\beta \cdot \frac{|F|^{2-\beta}}{|G_k|}
$$

with $\beta \in (0, 1)$. The dimensional balance (exponents sum to one) is what
makes the log-domain analysis tractable — see the paper for details.

## Citation

```bibtex
@misc{bernau2026overdrive,
  author       = {Bernau, Marc},
  title        = {Stability and convergence of the {Overdrive} variant
                  of the iterative {F}ourier transform algorithm},
  year         = {2026},
  eprint       = {arXiv:XXXX.XXXXX},
  archivePrefix= {arXiv}
}
```

## License

Apache License 2.0. See `LICENSE`.
