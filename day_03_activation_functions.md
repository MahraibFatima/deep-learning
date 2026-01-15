
# activation functions

a neural network without an activation function is just a giant linear regression model, no matter how many layers. an activation function is a non-linear transformation applied to input weights.

## sigmoid (logistic function)

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

range is $(0, 1)$. perfect for binary classification.

**flaws:**
- vanishing gradient.
- computationally expensive ($e^{-x}$).

**when to use today:**
- never in hidden layer.
- use in output layer for classification problems.

side note: dont surprise by the formulas representation or think it's AI generated. i have pretty good experience in latex/math pdf editing.
## hyperbolic tangent (tanh)

$$
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

range is $(-1, 1)$.

**betterment:** zero-centered, leading to faster convergence.

**flaw:**
- vanishing gradient with inputs of large magnitude.

**when to use:**
- sometimes in hidden layers of rnns/lstms.

## rectified linear unit (relu)

$$
\text{relu}(x) = \max(0, x)
$$

range is $[0, \infty)$.

**betterment:**
- solved vanishing gradient problem: derivative is 1 for $x > 0$, so gradient flows freely.
- computation is cheap.

**flaw:**
- dying relu.

**when to use:** 90% used in hidden layers. if it works, don't touch it.

## leaky relu

$$
\text{leakyrelu}(x) = \max(\eta x, x)
$$

**betterment:** provides a small, non-zero step for negative inputs, allowing neurons to recover.

**when to use:** if the "dying relu" problem occurs (check activation stats).

## exponential linear unit (elu)

$$
\text{elu}(x) = 
\begin{cases} 
x & \text{if } x \geq 0 \\
\eta(e^x - 1) & \text{otherwise}
\end{cases}
$$

## gaussian error linear unit (gelu)

$$
\text{gelu}(x) \approx 0.5x\left(1 + \tanh\left[\sqrt{\frac{2}{\pi}}\left(x + 0.044715x^3\right)\right]\right)
$$

**betterment:** default SOTA for transformers.

side note: i read my second research paper, but it was the first one i read from a learning perspective, so i'm happy about it. this formula was also copied from a research paper.

## swish (from google brain)

$$
\text{swish}(x) = x \cdot \sigma(\beta x)
$$

$\beta$ is often 1.

**when to use:** good alternative for relu, used for cnn tasks.
