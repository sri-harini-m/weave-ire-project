# Validation of Simulated Readers on MIND Dev Data

Evaluated across real user impressions in `data/raw/dev/behaviors.tsv`:

| Model                           |   Mean_AUC |    Std_Err |   CI_95_Lower |   CI_95_Upper |   Evaluated_Impressions |
|:--------------------------------|-----------:|-----------:|--------------:|--------------:|------------------------:|
| Random                          |   0.498871 | 0.00297762 |      0.493035 |      0.504707 |                   10000 |
| Popularity                      |   0.529774 | 0.00240985 |      0.525051 |      0.534498 |                   10000 |
| Simulated Reader Beliefs (Ours) |   0.646853 | 0.00292209 |      0.641126 |      0.652581 |                   10000 |
