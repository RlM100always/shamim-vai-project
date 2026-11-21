# Dataset

This project selects **Dataset option 1: Multi-Domain Sentiment Dataset (version 2.0)** from the assignment brief.

The official JHU page provides three variants:

- `processed_acl.tar.gz` (~19 MB): balanced binary-labelled sparse bag-of-words features. This is the default variant used here.
- `processed_stars.tar.gz` (~33 MB): preprocessed features with star-rating labels.
- `unprocessed.tar.gz` (~1.5 GB): original review records.

Download the default variant from the project root:

```bash
python scripts/download_dataset.py --variant processed_acl
```

The archive is extracted under `data/raw/processed_acl/`. Raw data are excluded from Git because they are large and redistribution terms should be checked with the provider. The repository contains a downloader, not a copy of the data.

Official source: https://www.cs.jhu.edu/~mdredze/datasets/sentiment/

The JHU page documents the preprocessed format as one document per line: `feature:<count> ... #label#:<label>`. Each directory corresponds to a domain. The parser also supports the common `positive.review` / `negative.review` layout.

The other three assignment options are documented in the assignment brief:

1. Diplomacy Dataset: https://sites.google.com/view/qanta/projects/diplomacy
2. Blog Authorship Corpus: https://huggingface.co/datasets/barilan/blog_authorship_corpus
3. Tennis post-match press conference transcripts: https://convokit.cornell.edu/documentation/tennis.html

Only one of the four assignment datasets is required; this project does not mix unrelated datasets.
