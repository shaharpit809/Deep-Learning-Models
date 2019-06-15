# DNN on MNIST

In this problem, I build a DNN with 5 layers each of 1024 hidden units and last layer consisting of 10 units with a softmax activation; and train it till a point such that 98% test accuracy is achieved. Once training was done, I do a feedforward step on 1000 test samples such that a 10-dim probability vector per sample is achieved. For each 10-d output vector, the dim with the maximum probability (which will eventually decide the class label). This is plotted in a 10x10 grid. 

I repeat the same procedure for the second last layer which gives a 1024-dim vector per sample. Then I randomly select 10 random dimensions out of 1024 dimensions. Again I plot it in a 10x10 grid. 

Then I perform feedforward on all the layers and then apply dimension reduction using t-Stochastic Neighbor Embedding (tSNE) and Principal Component Analysis (PCA) and visualize them to check their performance.
