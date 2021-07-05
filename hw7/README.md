I use `hfl/chinese-macbert-large` from Hugging Face pretrain model.

## Quick start
To run my code, simply first run the following three notebooks
* `hw7_macbert4.ipynb`          (Kaggle public: 0.85526)
* `hw7_macbert4_w_val.ipynb`    (Kaggle public: 0.83810)
* `hw7_macbert6.ipynb`          (Kaggle public: 0.84610)

These will generate model confiurations and checkpoints in folders `saved_model/macbert4`, `saved_model/macbert4_val` and
`saved_model/macbert6` that will be used in `ensemble_fusion859.ipynb`

The ensemble I used is fusion ensemble, which averages the three models' output logits.
* `ensemble_fusion859.ipynb`    (Kaggle public: 0.85926)

Final Kaggle public socre: 0.85926

References: https://huggingface.co/hfl/chinese-macbert-large