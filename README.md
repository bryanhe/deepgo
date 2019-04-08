Introduction
============

This repository mostly consists of third-party code related to
 * (1) [Move Evaluation in Go using Deep Convolutional Neural Networks](http://www.stats.ox.ac.uk/~cmaddis/pubs/deepgo.pdf)
 * (2) [Mastering the game of Go with deep neural networks and tree search](https://www.nature.com/articles/nature16961.pdf)
 * (3) [Mastering the game of Go without human knowledge](https://www.nature.com/articles/nature24270.pdf)

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
python -m deepgo train --batch 512 --checkpoint checkpoint --keep_all_checkpoints --workers 8
```

Prototype C++ Version
=====================

04/08/2019
In deepgo/cpp/external/
wget https://download.pytorch.org/libtorch/nightly/cu100/libtorch-shared-with-deps-latest.zip

rm -rf build
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/home/bryanhe/deepgo/cpp/external/libtorch ..
make

https://discuss.pytorch.org/t/libtorch-cmake-issues/28246
grep culibos `find .`

https://github.com/pytorch/examples/pull/506
