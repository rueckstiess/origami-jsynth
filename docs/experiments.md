# Experiment Reproduction


### Tabdiff on Yelp

```
origami-jsynth all --dataset yelp --model tabdiff -R 10 --param batch_size=512 --param sample_batch_size=512 --param check_val_every=20 --max-minutes 1440
```

### Tabdiff on DDXPlus

Train

origami-jsynth train --dataset ddxplus --model tabdiff -R 10 --param batch_size=512 --param check_val_every=20 --max-minutes 1440


Sample 

origami-jsynth sample --dataset ddxplus --model tabdiff -R 10

