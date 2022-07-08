# ExpressionGAN

### Controlling gene expression with deep generative design of regulatory DNA

<img src="https://github.com/JanZrimec/ExpressionGAN/blob/main/docs/fig3c.png" width="600">

Figure. Predictor-guided generator optimization enables gene-specific navigation of the regulatory sequence-expression landscape. T-distributed stochastic neighbor embedding (t-SNE) mapping of the input latent subspaces that produce novel sequence variants spanning ~6 orders of magnitude of gene expression (colored and black dots), uncovered using the predictor-guided generator optimization. Black dots represent selections of 10 sequence variants per each of the 4 expression groups covering a 4 order-of-magnitude range of predicted expression levels from TPM ~10 to ~10,000.

------------

Scripts for training and optimization of ExpressionGAN as well as to reproduce the analysis are provided in the folder 'scripts'.

The data including generated sequence data are available at [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6811226.svg)](https://doi.org/10.5281/zenodo.6811226), extract the archive to a folder named 'data'.

Software dependencies are specified in the environment files in the 'docs' folder, with env_training.yml used for GAN training and optimization and env_analysis.yml used for the data analysis.
