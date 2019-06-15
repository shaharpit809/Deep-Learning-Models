# Network Compression using Singular Value Decomposition (SVD)

In this problem, I train in DNN with 5 hidden layers of 1024 hidden units each and 10 units in the last layer; till a point such a that 98% accuracy is achieved. As we know, we can use SVD to compress the weights matrices. We will have 6 different weight matrices in the baseline network. 

First weight matrix shape : 784x1024 \
Next four weight matrices shape : 1024x1024 \
Last weight matrix shape : 1024x10

We will run SVD on the first 5 matrices because they are very large; to approximate the weight matrices such as:
![](https://github.com/shaharpit809/Deep-Learning-Models/blob/master/img/SVD_eqn1.PNG)

We will vary the value of D from 10, 20, 50, 100, 200, to Dfull in the below equation:
![](https://github.com/shaharpit809/Deep-Learning-Models/blob/master/img/SVD_eqn2.PNG)

where Dfull is the original size. Each of the 6 networks are using one of the 6 D values. We will calculate the accuracy of all the networks and check the performance of our networks. By only selecting D singular values, we can do some compression. 

Since the performance of the network is low in case of small values of D, we fix the value of D to be 20 and use the previous weights and train the network again on only 20 singular values. 

This way I was able to acheive good amount of accuracy and the compressed network only uses about 4% of the memory of the original network.