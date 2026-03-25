# Data

This directory is intentionally empty in the repository. The training dataset is too large to host on GitHub (~4.7 GB).

## Download the ASL Alphabet Dataset

```bash
python src/training/downloadataset.py
```

This downloads the dataset from Kaggle into `data/ASL_Alphabet_Dataset/` using `kagglehub`. You will need valid Kaggle API credentials (`~/.kaggle/kaggle.json`).

Alternatively, download it manually from Kaggle and place it at:

```
data/
└── ASL_Alphabet_Dataset/
    ├── A/
    ├── B/
    ...
    └── Z/
```
