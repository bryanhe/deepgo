Introduction
============

This repository mostly consists of third-party code related to
 * [(1) Move Evaluation in Go using Deep Convolutional Neural Networks](http://www.stats.ox.ac.uk/~cmaddis/pubs/deepgo.pdf)
 * [(2) Mastering the game of Go with deep neural networks and tree search](https://www.nature.com/articles/nature16961.pdf)
 * [(3) Mastering the game of Go without human knowledge](https://www.nature.com/articles/nature24270.pdf)

Currently, the code implements supervised learning applied to high-level human play from the [KGS server](https://www.gokgs.com/) (data found [here](https://www.u-go.net/gamerecords/)).
This is mostly explored in (1) (although it is also used in (2)). The neural network used is a clone of the network from (3).

Data
====
To download the data, run
```
bin/download_kgs.sh
```

To setup the data for training, run
```
python -m deepgo prepare
```

Training
========
Run
```
python -m deepgo train
```
