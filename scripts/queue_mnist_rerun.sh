#!/bin/sh
# MNIST ablations with the final default configuration (no reconstruction term).
python3 scripts/suite.py --datasets mnist --methods ahr_lossless ahr_lossy_mini ahr_lossless_mini --seeds 0 1 2 --jobs 1 --threads 1
S="python3 scripts/suite.py --datasets mnist --methods ahr --seeds 0 1 2 --jobs 1 --threads 1"
$S --tag no_memorize --extra "--memorize-epochs 0"
$S --tag reencode --extra "--memory-mode reencode --alpha-mem 0 --memorize-epochs 0"
