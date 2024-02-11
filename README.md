# Contents:
1. Describtion
2. Problem
3. Method
    * 3.1. Model
    * 3.2. Computation Graph
    * 3.3. Back Propagation
    * 3.4. Gradient Descent
4. Code
5. Results

# 1. Describtion
Description of the Neural Network of FNN architecture to solve MNIST classification problem. This project is an exercise of the idea of gradient descend and offers only mathematical approach whereas provided program is a pseudo-code showing the concept.

# 2. Problem
Apparatus of stochastic gradient descent.

# 3. Method
Project describes the apparatus in 3 parts: Model, Computation Graph, Back Propagation and Gradient Descent. Describtion of every part is provided below.

## 3.1. Model
![alt text](https://github.com/AKAD0/FNN_MNIST/blob/master/Fig1.png)

$$
\text{Fig.1: Topology of the architecture}
$$

Topology consists of 2 hidden layers 10 blocks each, 1 output layer of 10 blocks and 1 input layer of 784 blocks (28*28 pixels of an image to process).

### 3.2. Computation Graph:
![alt text](https://github.com/AKAD0/FNN_MNIST/blob/master/Fig2.png)

$$
\text{Fig.2: Computation graph}
$$

$w$ and $b$ parameters are vectors of weights and biases respectively. $y$ is a label mark for $\hat{y}=a^{(y)}$ to compare to.

$$
~\\
~\\
C=MSE=\frac{1}{10}\sum_{i=1}^{10}(y_i-a_i^{(y)})^2
$$

$$
\begin{aligned}
\text{where}~
&C=MSE-\text{Mean Square Error cost function} \\
&y-\text{true label} \\
&a^{(y)}-\text{result of the activation function} \\
&i=10-\text{number of predicting classes}~f^{(1)} \\
\end{aligned}
$$

$$
~\\
~\\
a^{(y)}=softmax(z^{(y)})= \frac{e^{z_i^{(y)}}}{\sum_{j=1}^{10}e^{z_j^{(y)}}}
$$

$$
\begin{aligned}
\text{where}~
&a^{(y)}-\text{activation function of output layer (y)} \\
&z^{(y)}-\text{input function (affine transformation) of output layer (y)} \\
&j=10-\text{number of predicting classes}~f^{(1)} \\
\end{aligned}
$$

$$
~\\
~\\
z^{(y)}=\sum_{i=1}^{10}(w_i^{(y)T}a_i^{(2)})+b^{(y)T}
$$

$$
\begin{aligned}
\text{where}~
&z^{(y)}-\text{input function (affine transformation) of output layer (y)} \\
&w^{(y)}-\text{weights vector of output layer (y)} \\
&b^{(y)}-\text{biases vector of output layer (y)} \\
&a^{(2)}-\text{activation function of the 2nd hidden layer} \\
\end{aligned}
$$

$$
~\\
~\\
a^{(2)}=ReLU=max\{0,z^{(2)}\}=
\left\{ \begin{array}{l} 
z^{(2)}, z^{(2)}>0 \\
0, z^{(2)}≤0
\end{array} \right.
$$

$$
\begin{aligned}
\text{where}~
&a^{(2)}-\text{activation function of the 2nd hidden layer} \\
&z^{(2)}-\text{input function (affine transformation) of 2nd hidden layer} \\
\end{aligned}
$$

$$
~\\
~\\
z^{(2)}=\sum_{i=1}^{10}(w_i^{(2)T}a_i^{(1)})+b^{(2)T}
$$

$$
\begin{aligned}
\text{where}~
&z^{(2)}-\text{input function (affine transformation) of 2nd hidden layer} \\
&w^{(2)}-\text{weights vector of 2nd hidden layer} \\
&b^{(2)T}-\text{biases vector of 2nd hidden layer} \\
&a^{(1)}-\text{activation function of the 1st hidden layer} \\
\end{aligned}
$$

$$
~\\
~\\
a^{(1)}=ReLU=max\{0,z^{(1)}\}=
\left\{ \begin{array}{l} 
z^{(1)}, z^{(1)}>0 \\
0, z^{(1)}≤0
\end{array} \right.
$$

$$
\begin{aligned}
\text{where}~
&a^{(1)}-\text{activation function of the 1st hidden layer} \\
&z^{(1)}-\text{input function (affine transformation) of 1st hidden layer} \\
\end{aligned}
$$

$$
~\\
~\\
z^{(1)}=\sum_{i=1}^{784}(w_i^{(1)T}x_i)+b^{(1)T}
$$

$$
\begin{aligned}
\text{where}~
&z^{(2)}-\text{input function (affine transformation) of 2nd hidden layer} \\
&w^{(2)}-\text{weights vector of 2nd hidden layer} \\
&b^{(2)T}-\text{biases vector of 2nd hidden layer} \\
&a^{(1)}-\text{activation function of the 1st hidden layer} \\
\end{aligned}
$$

### 3.3. Back Propagation:
Back Propagation is a process of finding derivatives of the Cost function with respect to the Theta parameters - weights and biases.