README:
conda create -n bio python=3.11.13
pip install torch fair-esm biopython numpy scipy tqdm pandas joblib seaborn "fastcore<1.9,>=1.8.0" fastai==2.8.4
pip install -U scikit-learn==1.7.2



PCA: 4 pca were done: one on uniref, one on random sequences, one on a dataset with tonb mutations and one on all these datasets together. The datasets are present in /pca/datasets.

