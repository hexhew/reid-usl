# Self-paced Contrastive Learning with Hybrid Memory for Domain Adaptive Object Re-ID

## Introduction

```
@inproceedings{ge2020self,
  title={Self-paced Contrastive Learning with Hybrid Memory for Domain Adaptive Object Re-ID},
  author={Ge, Yixiao and Chen, Dapeng and Zhu, Feng and Zhao, Rui and Li, Hongsheng},
  booktitle={NeurIPS},
  year={2020}
}
```

## Results

**Unsupervised version of SpCL**

- Market-1501

|                  | mAP  | Rank-1 | Rank-5 | Rank-10 |
|------------------|------|--------|--------|---------|
| (paper)          | 73.1 | 88.1   | 95.1   | 97.0    |
| repro. (best)    | 72.4 | 88.7   | 95.8   | 97.2    |
| repro. (latest)  | 72.5 | 87.7   | 95.2   | 97.1    |
| repro. (k1 = 20) | 75.3 | 89.1   | 95.5   | 97.4    |

- DukeMTMC-reID
|                  | mAP  | Rank-1 | Rank-5 | Rank-10 |
|------------------|------|--------|--------|---------|
| repro. (best)    | 63.1 | 79.0   | 88.8   | 92.2    |
| repro. (latest)  | 63.0 | 78.6   | 88.9   | 92.5    |
