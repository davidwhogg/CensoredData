# CensoredData

How can we make the best use of non-detections and noisy data when we
don't believe the uncertainty or upper limit informaiton being
reported by the data provider?

### Authors:

* **Nat Butler** (ASU)
* **David W. Hogg** (NYU)
* **James Long** (Berkeley)
* **Joey Richards** (Berkeley)

### License:

Copyright 2011, 2012 the authors.  **All rights reserved**.

### Contents:

#### data:

* `mira_sims`: light curves for simulated mira variables.  LCs are of ASAS 235627-4947.2 with varying fractions of the original flux, e.g., `1, 0.5, 0.25, 0.01`

* `miras`: ASAS light curves for the 1720 mira candidates, with non-detections denoted with `29.999`

* `mira_features.dat`: features for 1720 mira candidates, including all Lomb-Scargle and non-LS features

* `rrl`: ASAS light curves for the 1029 RR Lyrae, FM candidates, with non-detections denoted with `29.999`

* `rrl_features.dat`: features for 1029 RR Lyrae, FM candidates, including all Lomb-Scargle and non-LS features.

#### plots:

`mira_simulated.pdf`: simulated mira light curve with different flux fractions

#### py:

the code

#### tex:

* `Makefile`: file to compile Hogg's original note

* `censored_catalog.tex`: tex for Hogg's original note

* `ms.tex`: manuscript tex file

* `non_detect.bib`: manuscript `bib` file

* `apj.bst`: needed for `ms.tex` to compile
