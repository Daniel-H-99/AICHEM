# Smile BERT Classifier of HIV Effectives
This is a repository for implentation of BERT based classifer working on SMILES.
Architecture resembles one explained in paper.
Several noticeable points are:
* Requirements: Python 3
* Dataset: HIV dataset (positive << negative)
* Features: Model, Framework agnostic
* hyperparemeters
   * epochs = 100
   * batch = 128
   * optimizer = Adam
   * learning_rate = 1e-5
   * portion_noise = 0.3
   * temperature = 0.1
* No dropout & schedulers are implemented
* In pretraining stage, remaining negative samples were used as unlabeled dataset, after augmentation using noise token. Unsupervised learning procedure is same as SimCLR.
* In training stage, in order to overcome overfitting, noise token was used for making augmented sequence.

# Result
Model Score
      
      Precision, Accuracy, AUROC, Model Score = 0.68, 0.67, 0.71, 1.35

Baseline Score 

      Precision, Accuracy, AUROC, Model Score = 0.61, 0.61, 0.65, 1.21

# Run

      sh run.sh

### References
[1] Transformer in DGL (https://github.com/dmlc/dgl/tree/master/examples/pytorch/transformer)

