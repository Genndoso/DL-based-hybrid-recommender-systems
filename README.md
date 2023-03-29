## Comparison study of DeepFM Hybrid vs. baselines

In this project, we implemented several baselines (LightFM, DSSM, and SVD) and DeepFM model. We trained our models on the [KION dataset](https://www.kaggle.com/datasets/asenin/kion-dataset). 

### Dataset Preprocessing:
We performed some preliminary preprocesing on the mentioned dataset. The notebook `data_preprocessing.ipynb` was used to procecss the data. After running this notebook, there should be 3 created `csv` files for `items`, `users`, and `interactions` respectively. These steps are enough to run our baselines (LightFM, DSSM, SVD)

For the data prepoccessing for DeepFM, the function `preproc_deep_fm` in `utils/preprocessing.py` were used. After running this function, it will combine the 3 `csv` files above to create 1 big `csv`. Alternatively, the preproccessed data can be downloaded [here](https://drive.google.com/file/d/15zfmD-qvnYPSFwPJwHUmvhQX62CDN00O/view?usp=sharing).

### Reproducing Results:

You can reproduce the results for the baselines in their respective notebooks (`LIGHFM.ipynb`, `SVD.ipynb`, and `DSSR.ipynb`). To reproduce the results for DeepFM, you can run the `main.py` script in `src/DeepFM/main.py`:

```
python3 main.py --train True --cold_start True
```

The `train` flag is to either train or validate. The `cold_start` flag is to either validate on cold start scenerio or standard scenerio. 


### References:
Guo H., Tang R., Ye Y., Li Z. DeepFM: An End-to-End Wide & Deep Learning Framework for CTR Prediction. 2015

