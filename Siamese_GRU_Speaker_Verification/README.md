# Siamese Network using a GRU model for Speaker Verification

In this problem, I perform speaker verification on audio clips using a [Siamese Network](https://en.wikipedia.org/wiki/Siamese_network) . The training file(trs.pkl) consists of a [500 x 16180] matrix, whose row is a speech signal with 16,180 samples. They are the returned vectors from the librosa.load function. Similarly, the test file(tes.pkl) holds a [200 x 22631] matrix.

The training matrix is ordered by speakers. Each speaker has 10 utterances, and there are 50 such speakers (that's why there are 500 rows). Similarly, the test set has 20 speakers, each of which is with 10 utterances.

I then randomly sample L pairs of utterances from the ten utterance of the first speaker. In theory, there are total 45 pairs(using combination formula : 10C2)). These are the positive examples in the first minibatch. Similarly from the other 49 speakers, I take L pairs with the ten utterances of the first speaker. These are the negative examples in the first minibatch.

Now the first minibatch is of size 2L and L positive and negative pairs.

I repeat this process for other 49 speakers and get 50 minibatches with a balanced number of positive and negative pairs.

Then I create a Siamese Network using 2 GRU cells with 2 fully connected layers. The tries to predict 1 for the positive pairs and 0 for the negative ones.

Similarly, I tested the model on the test file(tes.pkl) and get an accuracy of 68.33%.
