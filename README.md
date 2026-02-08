INSTRUCTIONS ON HOW TO CORRECTLY USE THIS REPO:

Preliminary step: Run the following lines to create the required environment:
conda create -n bio python=3.11.13
pip install torch fair-esm biopython numpy scipy tqdm pandas joblib seaborn "fastcore<1.9,>=1.8.0" fastai==2.8.4
pip install -U scikit-learn==1.7.2

FOLDERS:

sequences:
has the txt files with the sequences of TonB and other proteins inside.

layer_selection: 
The folder contains a script that will create a number of conservative and non conservative mutations, then will give the ratio between the cosine similarities in embeddings between TonB and mutants, for each layer from 20 to 33. The number of layers has to be chosen in order not to be in a highly anisotropic space (such as the one after layer 33), and has also to be a significative layer, so not too close to the internal ones. For reference, layer 28 was used for the TonB analysis. 

cosine_similarity:
Creates histograms of the cosine similarity, for different number of TonB mutations. (Uses mean pooling!)

pca: 
4 PCAs were done: one on an uniref50 subsample, one on random sequences, one on a dataset with tonb mutations and one on all these datasets together (50k each). The datasets are present in ESM_analyzer/pca/datasets.
Datasets: an important step has to be done for the uniref dataset: it has to be downloaded from ......., and the file has to be manually moved from the computer's downloads to the ESM_analyzer/pca/datasets folder. The reason for this is that github has a limit on the max file dimension, and the uniref subsample exceeds it. The file path HAS to be /pca/datasets/uniref50_subsample.fasta , in order for the gitignore file to be able to not include it in a future push on github. Spiega ancora come funziona ecc ecc FEDE E CATA: TESTARE I FILE DELLE PCA E CAPIRE COME ADATTARE I FILE

segment_pooling: 
Mean pooling on the residue embeddings was discarded as an option, since it compresses all the information and projects all the embeddings in a very narrow cone in the embedding space. Thus, a segment-based pooling technique was used: the residues are embedded, then the residue embeddings are grouped together in segments (POSSIBLE FUTURE IDEA: segmentation of the sequence with respect to the domains), then the mean pooling is done within the segments, then they are normalized. The cosine similarity between 2 sequences is then given by the MEAN of the cosine similarities of corresponding SEGMENTS. The file test.py can be used to test if the environment and the segment pooling works, while segment_number_selection.py looks at how far apart are conservative and non conservative mutations, for different segment numbers. Small number of segments -> more averaging and less local information. High number of segments -> more info on local features, but also more noise.

markov:

When running the scripts for the first time, the esm model has to be downloaded. Later, it will be stored in .cache and be always available. It could take like 15 mins to download.

