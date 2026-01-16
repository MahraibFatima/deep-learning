yesterday while learning about activation functions. we came across 2 distinguished terms. 
1. vashing gradient
2. dying relu

here is a short summry of these. 

### vanishing gradient
vanishing gradient is a problem that happens during training in deep neural networks, especially those using activation functions like `sigmoid` or `tanh`.
what happens?

during `backpropagation`, gradients (derivatives) are calculated and passed backward through the network. these gradients tell the model how much to adjust each weight to reduce error.

with certain activation functions, the gradient can become extremely small close to zero, as it gets multiplied layer by layer.

if the gradient becomes too small, the weights in earlier layers receive almost no update, so they stop learning.

---

why does it happen?

for example, the derivative of `sigmoid` is:

$\sigma'(x) = \sigma(x)(1 - \sigma(x))$

since $\sigma(x)$ is between 0 and 1, the derivative is between 0 and 0.25.
if you multiply many small numbers (like $0.1 \times 0.1 \times 0.1$...), the result approaches zero very quickly.

`tanh` has a similar problem: its derivative is between 0 and 1, but for large inputs it also saturates and gives near-zero gradients.

---

the result:

· early layers learn very slowly or not at all. 

· deep networks become hard or impossible to train. 

---

how is it solved?

modern activation functions like `relu` help because:

· for x > 0, derivative is exactly 1, so gradients don’t shrink
· no saturation in the positive region

but `relu` introduces its own problem: `dying relu`, where neurons can get stuck at zero and also stop learning.
variants like `leaky relu`, `elu`, and `gelu` try to fix this while keeping gradients learning.

---

### dying relu
`dying relu` is a problem that happens when neurons using the `relu` activation function become permanently "dead", meaning they stop firing or outputting zero for all inputs and never recover.

---

what happens?

a `relu` neuron outputs:

$\text{relu}(x) = \max(0, x)$

this means:

· if the weighted sum  x  is positive → output =  x 
· if  x  is negative → output = 0

the derivative for:

·  x > 0  → 1
·  x < 0  → 0

---

how do neurons die?

during training, if a neuron's weighted sum becomes negative for all training examples, its gradient becomes 0 (because derivative is 0 for negative inputs).

once the gradient is 0, the weights won’t update → the neuron stays "off" forever → it's dead.

this is especially common if:

· learning rate is too high. 

· large weight updates push the neuron into negative territory permanently. 

· bad weight initialization. 

---

why is it a problem?

dead neurons don’t contribute to learning, they’re wasted parameters. too many dead neurons can reduce the network’s capacity and slow learning.

---

how to fix it?

use variants of `relu` that allow a small gradient for negative inputs:

`leaky relu`:

```python
def leaky_relu(x, alpha=0.01):
    return np.maximum(alpha * x, x)
```

→ small slope (alpha) for negatives, so gradient never fully dies.

parametric relu (prelu):
like leaky `relu`, but alpha is learned.

`elu`:

```python
def elu(x, alpha=1.0):
    return np.where(x >= 0, x, alpha * (np.exp(x) - 1))
```

- smooth for negatives, helps mean activations stay closer to zero.



if you came this far.. thanks for reading. ✨
