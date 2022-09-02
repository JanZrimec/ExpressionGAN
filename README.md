# ExpressionGAN

### Controlling gene expression with deep generative design of regulatory DNA
Link to paper: [10.1038/s41467-022-32818-8](https://doi.org/10.1038/s41467-022-32818-8)

<img src="https://github.com/JanZrimec/ExpressionGAN/blob/main/docs/fig3c.png" width="600">

Figure. Predictor-guided generator optimization enables gene-specific navigation of the regulatory sequence-expression landscape. T-distributed stochastic neighbor embedding (t-SNE) mapping of the input latent subspaces that produce novel sequence variants spanning ~6 orders of magnitude of gene expression (colored and black dots), uncovered using the predictor-guided generator optimization. Black dots represent selections of 10 sequence variants per each of the 4 expression groups covering a 4 order-of-magnitude range of predicted expression levels from TPM ~10 to ~10,000.

------------
### Note
Arrows were incorrectly rendered and are missing in schematic figures 1d, 3a & 6e. The correct panels are available in the docs folder.

<img src="https://github.com/JanZrimec/ExpressionGAN/blob/main/docs/fig1d.png" width="400"> 

Figure 1d. Overview of the generative adversarial network (GAN) approach.


<img src="https://github.com/JanZrimec/ExpressionGAN/blob/main/docs/fig3a.png" width="400">

Figure 3a. Schematic depiction of the procedure to optimize the generator.


<img src="https://github.com/JanZrimec/ExpressionGAN/blob/main/docs/fig6e.png" width="400">

Figure 6e. Schematic depiction of the mutagenesis strategy.


------------
Scripts for training and optimization of ExpressionGAN as well as to reproduce the analysis are provided in the folder 'scripts'.

The data including generated sequence data are available at [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6811226.svg)](https://doi.org/10.5281/zenodo.6811226), extract the archive to a folder named 'data'.

Software dependencies are specified in the environment files in the 'docs' folder, with env_training.yml used for GAN training and optimization and env_analysis.yml used for the data analysis.
