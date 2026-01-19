## forward propogation (FP)

while reading & learning about FP, i came across many definition and found this simple and exact definition.
> Forward propagation is where input data is fed through a network, in a forward direction, to generate an output. The data is accepted by hidden layers and processed, as per the activation function, and moves to the successive layer. The forward flow of data is designed to avoid data moving in a circular motion, which does not generate an output. 

[reference](https://h2o.ai/wiki/forward-propagation/)

### math represtentation:
-> for `l` layers:
```
Z⁽ˡ⁾ = W⁽ˡ⁾ · A⁽ˡ⁻¹⁾ + b⁽ˡ⁾
A⁽ˡ⁾ = f(Z⁽ˡ⁾)
```
-> for one neuron porward pass:
```
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
a = f(z)
```
> where `f` is activation function.

<img width="1249" height="666" alt="image" src="https://github.com/user-attachments/assets/ddc147d8-f3f1-4b09-85be-a8e0eadb013d" />


here is simple code for forward propogation, with 2 layers + output layer

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def forward_propagation(X, parameters, activation='relu'):

    #layer 1
    Z1 = np.dot(parameters['W1'], X) + parameters['b1']
    A1 = relu(Z1)

    #layer 2  
    Z2 = np.dot(parameters['W2'], A1) + parameters['b2']
    A2 = relu(Z2)

    #output layer
    Z3 = np.dot(parameters['W3'], A2) + parameters['b3']
    A3 = sigmoid(Z3)
    
    cache = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2, 'Z3': Z3, 'A3': A3}
    
    return A3, cache
```
### benefit of forward propogation
- easy computation: just matrix multiplications and element-wise operations.

### why need back propogation?

1. no learning: bcz forward propogation only computes predictions, doesn't update weights.

2. forward pass only: forward propogation doesn't tell us how wrong we are. to determine the how bad the prediction of NN is, we need to compute loss functions & update the weights.

