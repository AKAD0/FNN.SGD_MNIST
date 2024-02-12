# Contents:
1. Describtion
2. Problem
3. Method
    * 3.1. Model
    * 3.2. Computation Graph
    * 3.3. Back Propagation
    * 3.4. Stochastic Gradient Descent
4. Code
5. Results

# 1. Describtion
Description of the Neural Network of FNN architecture to solve MNIST classification problem. This project is an exercise of the idea of gradient descend and offers only mathematical approach whereas provided program is a pseudo-code showing the concept.

# 2. Problem
Apparatus of stochastic gradient descent.

# 3. Method
Project describes the apparatus in 3 parts: Model, Computation Graph, Back Propagation and Gradient Descent. Describtion of every part is provided below.

## 3.1. Model
<p align="center">
  <img src="https://github.com/AKAD0/FNN_MNIST/blob/master/Fig1.png">
</p>

$$
\text{Fig.1: Topology of the architecture}
$$

Topology consists of 2 hidden layers 10 blocks each, 1 output layer of 10 blocks and 1 input layer of 784 blocks (28*28 pixels of an image to process).

### 3.2. Computation Graph:
<p align="center">
  <img src="https://github.com/AKAD0/FNN_MNIST/blob/master/Fig2_v2.png">
</p>

$$
\text{Fig.2: Computation graph}
$$

$w$ and $b$ parameters are vectors of weights and biases respectively. $y$ is a label mark for $\hat{y}=a^{(y)}$ to compare to.

‎<br>

$$
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

‎<br>

$$
a^{(y)}=softmax(z^{(y)})= \frac{e^{z_i^{(y)}}}{\sum\limits_{j=1}^{10} e^{z_j^{(y)}}}
$$

$$
\begin{aligned}
\text{where}~
&a^{(y)}-\text{activation function of output layer (y)} \\
&z^{(y)}-\text{input function (affine transformation) of output layer (y)} \\
&j=10-\text{number of predicting classes}~f^{(1)} \\
\end{aligned}
$$

‎<br>

$$
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

‎<br>

$$
a^{(2)}=ReLU=max\lbrace 0,z^{(2)}\rbrace=
\begin{Bmatrix}
  z^{(2)},~z^{(2)}>0 \\
  0,~z^{(2)}≤0
\end{Bmatrix}
$$

$$
\begin{aligned}
\text{where}~
&a^{(2)}-\text{activation function of the 2nd hidden layer} \\
&z^{(2)}-\text{input function (affine transformation) of 2nd hidden layer} \\
\end{aligned}
$$

‎<br>

$$
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

‎<br>

$$
a^{(1)}=ReLU=max\lbrace 0,z^{(1)}\rbrace=
\begin{Bmatrix}
  z^{(1)},~z^{(1)}>0 \\
  0,~z^{(1)}≤0
\end{Bmatrix}
$$

$$
\begin{aligned}
\text{where}~
&a^{(1)}-\text{activation function of the 1st hidden layer} \\
&z^{(1)}-\text{input function (affine transformation) of 1st hidden layer} \\
\end{aligned}
$$

‎<br>

$$
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
It is done in two steps:
1) Find all the derivatives in the computational graph for further use in the chain rule.
2) Find said 'Theta' derivatives via chain rule.

#### First step. Finding every derivative in the computational graph.
$$
\begin{aligned}
\text{1)}~\frac{dC}{da^{(y)}}~
&=\bigg| \frac{d\frac{1}{10}((y_1-a_1)^2+(y_2-a_2)^2+...+(y_10-a_10)^2)}{da_1} \\
&= -\frac{1}{10}2(y_1-a_1)=\frac{-2(y_1-a_1)}{10}=\frac{-(y_1-a_1)}{5} \bigg|\\
&=\left[ \frac{-(y_1-a_1)}{5},~...~,\frac{-(y_10-a_10)}{5} \right]^T\\
&=\frac{-(y-a^{(y)})}{5} \\
\end{aligned}
$$

‎<br>

$$
\begin{aligned}
\text{2)}\frac{da^{(y)}}{dz^{(y)}}=DZ^{(y)}_{R^{10}}
&=\left| \text{Quotient rule} \right| \\
&=
\left[ 
\begin{matrix} 
a^{(y)}_1(1-a^{(y)}_1), & a^{(y)}_2a^{(y)}_1,     & ~...~,  & a^{(y)}\_{10}a^{(y)}_1 \\
a^{(y)}_1a^{(y)}_2,     & a^{(y)}_2(1-a^{(y)}_2), & ~...~,  & a^{(y)}\_{10}a^{(y)}_2 \\
~...~,                  & ~...~,                  & ~...~,  & ~...~ \\
a^{(y)}_1a^{(y)}\_{10},  & a^{(y)}_2a^{(y)}\_{10},  & ~...~,  & a^{(y)}\_{10}(1-a^{(y)}\_{10})
\end{matrix}
\right] \\
&=
\left[
\begin{matrix}
\sum\limits\_{k=1}\^{10} (a^{(y)}_ka^{(y)}_1)-a^{(y)}_1a^{(y)}_1+a^{(y)}_1(1-a^{(y)}_1) \\
~...~ \\
\sum\limits\_{k=1}\^{10} (a^{(y)}_ka^{(y)}\_{10})-a^{(y)}\_{10}a^{(y)}\_{10}+a^{(y)}\_{10}(1-a^{(y)}\_{10})
\end{matrix}
\right] 
\end{aligned}
$$

‎<br>

$$
\text{3)}~\frac{dz^{(y)}}{da^{(2)}}=w^{(y)}
$$

‎<br>

$$
\text{4)}~\frac{da^{(2)}}{dz^{(2)}}= 
\begin{Bmatrix}
  1,~z^{(2)}>0 \\
  0,~z^{(2)}≤0
\end{Bmatrix}
\implies
𝟙_{R>0}(z^{(2)})
$$

‎<br>

$$
\text{5)}~\frac{dz^{(2)}}{da^{(1)}}=w^{(2)}
$$

‎<br>

$$
\text{6)}~\frac{da^{(1)}}{dz^{(1)}}= 
\begin{Bmatrix}
  1,~z^{(1)}>0 \\
  0,~z^{(1)}≤0
\end{Bmatrix}
\implies
𝟙_{R>0}(z^{(1)})
$$

‎<br>

$$
\text{7)}~\frac{dz^{(y)}}{dw^{(y)}}=a^{(2)}
$$

‎<br>

$$
\text{8)}~\frac{dz^{(y)}}{db^{(y)}}=1
$$

‎<br>

$$
\text{9)}~\frac{dz^{(2)}}{dw^{(2)}}=a^{(1)}
$$

‎<br>

$$
\text{10)}~\frac{dz^{(2)}}{db^{(2)}}=1
$$

‎<br>

$$
\text{11)}~\frac{dz^{(1)}}{dw^{(1)}}=x
$$

‎<br>

$$
\text{12)}~\frac{dz^{(1)}}{db^{(1)}}=1
$$

#### Second step. Finding 'Theta' derivatives via Chain rule.
$$
\text{1)}~\frac{dC}{dw^{(y)}} = \frac{dC}{da^{(y)}} \frac{da^{(y)}}{dz^{(y)}} \frac{dz^{(y)}}{dw^{(y)}}
$$

‎<br>

$$
\text{2)}~\frac{dC}{db^{(y)}} = \frac{dC}{da^{(y)}} \frac{da^{(y)}}{dz^{(y)}} \frac{dz^{(y)}}{db^{(y)}}
$$

‎<br>

$$
\text{3)}~\frac{dC}{dw^{(2)}} = \frac{dC}{da^{(y)}} \frac{da^{(y)}}{dz^{(y)}} \frac{dz^{(y)}}{da^{(2)}} \frac{da^{(2)}}{dz^{(2)}} \frac{dz^{(2)}}{dw^{(2)}}
$$

‎<br>

$$
\text{4)}~\frac{dC}{db^{(2)}} = \frac{dC}{da^{(y)}} \frac{da^{(y)}}{dz^{(y)}} \frac{dz^{(y)}}{da^{(2)}} \frac{da^{(2)}}{dz^{(2)}} \frac{dz^{(2)}}{db^{(2)}}
$$

‎<br>

$$
\text{5)}~\frac{dC}{dw^{(1)}} = \frac{dC}{da^{(y)}} \frac{da^{(y)}}{dz^{(y)}} \frac{dz^{(y)}}{da^{(2)}} \frac{da^{(2)}}{dz^{(2)}} \frac{dz^{(2)}}{da^{(1)}} \frac{da^{(1)}}{dz^{(1)}} \frac{dz^{(1)}}{dw^{(1)}}
$$

‎<br>

$$
\text{6)}~\frac{dC}{db^{(1)}} = \frac{dC}{da^{(y)}} \frac{da^{(y)}}{dz^{(y)}} \frac{dz^{(y)}}{da^{(2)}} \frac{da^{(2)}}{dz^{(2)}} \frac{dz^{(2)}}{da^{(1)}} \frac{da^{(1)}}{dz^{(1)}} \frac{dz^{(1)}}{db^{(1)}}
$$

‎<br>

$$
\begin{aligned}
&S_1 = \frac{dC}{da^{(y)}} \frac{da^{(y)}}{dz^{(y)}} = \frac{-(y-a^{(y)})}{5}DZ^{(y)} \\
&S_2 = S_1 \frac{dz^{(y)}}{da^{(2)}} \frac{da^{(2)}}{dz^{(2)}} = S_1 w^{(y)} 𝟙_{R>0}(z^{2}) \\
&S_3 = S_1 S_2 \frac{dz^{(2)}}{da^{(1)}} \frac{da^{(1)}}{dz^{(1)}} = S_1 S_2 w^{(2)} 𝟙_{R>0}(z^{1}) \\
\end{aligned}
$$

‎<br>

$$
\begin{aligned}
&\nabla _{w{(y)}} C = \frac{dC}{dw^{(y)}} = S_1a^{(2)} \\
&\nabla _{b{(y)}} C = \frac{dC}{db^{(y)}} = S_1 \\
&\nabla _{w{(2)}} C = \frac{dC}{dw^{(2)}} = S_1 S_2 a^{(1)} \\
&\nabla _{b{(2)}} C = \frac{dC}{db^{(2)}} = S_1 S_2 \\
&\nabla _{w{(1)}} C = \frac{dC}{dw^{(1)}} = S_1 S_2 S_3 x \\
&\nabla _{b{(1)}} C = \frac{dC}{db^{(1)}} = S_1 S_2 S_3 \\
\end{aligned}
$$

### 3.4. Stochastic Gradient Descent:
Gradient Descent in essence is a process of repeated steps of subtractions from current Theta parameters the mean value over every found gradients and multiplied with an epsilon coefficient to set learning 'rate', thus approximating to the local minimum of the Cost function.

Stochastic Gradient Descent modifies vanilla approach in order to drastically accelerate computation via dividing the whole dataset into randomly (stochastically) ordered 'mini-batches' where a processing of a single batch constitutes single step. Since mini-batch is way smaller than a whole dataset, it's processing takes less time, however with less precision.

$$
\begin{aligned}
&\nabla _{w{(y)}}\^{*} C = \frac{1}{K} \sum\limits\_{i=1}\^{K} \nabla _{w{(y)}i} C \\
&\nabla _{b{(y)}}\^{*} C = \frac{1}{K} \sum\limits\_{i=1}\^{K} \nabla _{b{(y)}i} C \\
&\nabla _{w{(2)}} C = \frac{1}{K} \sum\limits\_{i=1}\^{K} \nabla _{w{(2)}i} C \\
&\nabla _{b{(2)}} C = \frac{1}{K} \sum\limits\_{i=1}\^{K} \nabla _{b{(2)}i} C \\
&\nabla _{w{(1)}} C = \frac{1}{K} \sum\limits\_{i=1}\^{K} \nabla _{w{(1)}i} C \\
&\nabla _{b{(1)}} C = \frac{1}{K} \sum\limits\_{i=1}\^{K} \nabla _{b{(1)}i} C \\
\end{aligned}
$$

‎<br>

$$
\begin{aligned}
$w^{(y)}=w^{(y)}-\epsilon\nabla ^{*}_{w{(y)}} C
$b^{(y)}=b^{(y)}-\epsilon\nabla ^{*}_{b{(y)}} C
$w^{(2)}=w^{(2)}-\epsilon\nabla ^{*}_{w{(2)}} C
$b^{(2)}=b^{(2)}-\epsilon\nabla ^{*}_{b{(2)}} C
$w^{(1)}=w^{(1)}-\epsilon\nabla ^{*}_{w{(1)}} C
$b^{(1)}=b^{(1)}-\epsilon\nabla ^{*}_{b{(1)}} C
\end{aligned}
$$