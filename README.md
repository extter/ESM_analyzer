README:
conda create -n bio python=3.11.13
pip install torch fair-esm biopython numpy scipy tqdm pandas joblib seaborn "fastcore<1.9,>=1.8.0" fastai==2.8.4
pip install -U scikit-learn==1.7.2



PCA: 4 pca were done: one on uniref, one on random sequences, one on a dataset with tonb mutations and one on all these datasets together. The datasets are present in /pca/datasets.

Datasets: an important step has to be done for the uniref dataset: it has to be downloaded from ......., and the file has to be manually moved from the computer's downloads to the datasets folder. The reason for this is that github has a limit on the max file dimension, and the uniref subsample exceeds it. The file path has to be /pca/datasets/uniref50_subsample.fasta , in order for the gitignore file to be able to not include it in a future push on github. 

When running the scripts for the first time, the esm model has to be downloaded. Later, it will be stored in .cache and be always available. It could take like 15 mins to download.

FEDE E CATA: TESTARE I FILE DELLE PCA E CAPIRE COME ADATTARE I FILE