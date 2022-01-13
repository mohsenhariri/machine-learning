

1. define model
2. define hyper parameters:

- batch size
- number of epochs
- learning rate

3. initial parameters (weights and biases and ???)
   https://pytorch.org/docs/stable/nn.init.html

4. define optimizer algorithms
   Gradient Descent Adam, ...
5. define cost function (loss function)

conv2d:
input: tensor (minibatch_size, in_channels, H, W)

# Convolution

Input = (I<sub>H</sub>, I<sub>W</sub>)

Kernel = (K<sub>H</sub>, K<sub>W</sub>)

Stride = (S<sub>H</sub>, S<sub>W</sub>)

Padding = padH, padW

$$Output_H = \frac{I_H + 2 * padH - K_H}{S_H} + 1$$

$$Output_W = \frac{I_W + 2 * padW - K_W}{S_W} + 1$$
