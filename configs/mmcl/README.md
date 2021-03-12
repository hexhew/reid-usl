# Unsupervised Person Re-identification via Multi-label Classification

## Introduction

```
@inproceedings{wang2020unsupervised,
  title={Unsupervised person re-identification via multi-label classification},
  author={Wang, Dongkai and Zhang, Shiliang},
  booktitle={CVPR},
  pages={10981--10990},
  year={2020}
}
```

### Results

**MMCL without CamStyle**

- Market-1501

|         | mAP    | Rank-1   | Rank-5   | Rank-10   |
|---------|--------|----------|----------|-----------|
| (paper) | 35.3   | 66.6     | -        | -         |
| repro.  | 39.2   | 67.5     | 79.5     | 83.8      |

- DukeMTMC-reID

|         | mAP    | Rank-1   | Rank-5   | Rank-10   |
|---------|--------|----------|----------|-----------|
| (paper) | 36.3   | 58.0     | -        | -         |
| repro.  | 40.1   | 59.6     | 71.6     | 76.2      |

- MSMT17

|         | mAP    | Rank-1   | Rank-5   | Rank-10   |
|---------|--------|----------|----------|-----------|
| repro.  | 5.0    | 15.3     | 21.9     | 25.8      |
