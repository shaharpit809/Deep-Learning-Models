# Variational Autoencoders

In this problem, I implemented a Variational Autoencoders(VAE) to find out the effects added to the 7's. tr7.pkl contains 6,265 MNIST digits from its training set which has a rank-3 tensor of size [6,265 x 28 x 28]. Similarly, te7.pkl contains 1,028 7's.

VAE helps us find out a few latent dimensions, one of which should shows the effect added. VAE needs a hidden layer that is dedicated to learn the
latent embedding. In this layer, each hidden unit is governed by a standard normal distribution as its a priori information. I fixed the number of hidden units (K) to 3 to reduce the search space and also find the effect.

To check if the VAE model was performing fine or not and check if the latent dimensions are correct, I generated new 7's by feeding a few randomly
generated code vectors, that are the random samples from the K normal distributions that VAE learned.

Lastly, to check which dimension takes care of the effect added, I fixed [K-1] dimensions with the same value over the codes, while varying only one of them. I repeated this for all the dimensions and generated images to find out the effect.