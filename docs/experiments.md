# Experiment Reproduction


### Tabdiff on Yelp

```
origami-jsynth all --dataset yelp --model tabdiff -R 10 --param batch_size=512 --param sample_batch_size=512 --param check_val_every=20 --param max-minutes 1440
```