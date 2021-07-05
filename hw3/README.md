# Quick start to run my code

* Run `model1.ipynb`, `model2.ipynb`, `model3.ipynb`, `model4.ipynb`, respectively.

This will generate `model1.ckpt`, `model2.ckpt`, `model3.ckpt`, `model4.ckpt` and the corresponding predictions `model1.csv`, `model2.csv`, `model3.csv`,  `model4.csv`. Note that you must execute these models orderly, since some model is
retrained from previous models using unlabeled data (semi-supervised). For each model, the training time is around 6 to 8 hours on my gpu 1080ti.

* Run `Ensemble1.ipynb`.

This will ensemble model1, model2, model3 by voting. The resulting predictions are saved in `Ensemble1.csv` (kaggle public 0.81421)

* Run `Ensemble2.ipynb`.

This will ensemble model2, model3, model4 by fusion (averaged output from all models). The resulting predictions are saved in `Ensemble2.csv`
(kaggle public 0.81899)

* Run `Ensemble3.ipynb`.

This will ensemble model2, model3, model4 by voting. The resulting predictions are saved in `Ensemble3.csv`
(kaggle public 0.82795)

* Run `Ensemble4.ipynb`.

This will ensemble all the above three emsembles (Ensemble1, Ensemble2, Ensemble3) by voting. The resulting predictions are saved in `Ensemble4.csv`
(kaggle public 0.83094)
