# Adaptive Loss Scaling for Splitting Mods


__Expert:__ Aleksandr Beznosikov

__Consultant:__ Igor Ignashin


## Abstract

In machine learning, numerous challenges can degrade model performance, including noisy features and incorrect labeling in the training data. Various approaches exist to mitigate these issues, such as Adaptive Loss Scaling. In this paper, we propose an improvement to this approach by incorporating a label correction mechanism. Our method replaces the labels of high-loss samples with those of their nearest neighbors in the embedding space. We demonstrate the effectiveness of our approach on the MNIST dataset with a fraction of noisy labels.
