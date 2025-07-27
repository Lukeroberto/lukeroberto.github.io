# Learnings on the Muon optimizer


I have seen this optimizer pop up in several places now, but the most striking place for me was on the nanogpt
[benchmark](https://github.com/KellerJordan/modded-nanogpt). I would like to do a deeper dive soon on how these
"speedrun"-style benchmarks work, I'll just focus on the small rabbit hole I went in on the optimizer itself. A lot of
these learnings and analysis are from this post by [Jeremy Bernstein](https://jeremybernste.in/writing/deriving-muon).

The form of the optimizer is:

$$
W \leftarrow W - lr * \sqrt{\frac{f_{out}}{f_{in}}} * NewtonShulz(\nabla_W L)
$$


This looks pretty close to our usual formula for gradient descent, but with 2 notable differences. Where does the $\sqrt{\frac{f_{out}}{f_{in}}}$ come frome? And why this there this "Newton Shulz" operation happening after the gradient
computation?

## Linear Layers

Nearly all modern deep learning architectures will contain linear (dense) layers somewhere inside of them. They are the
simplest way to build connections from every neuron to every other neuron. 

When taking a gradient across a linear layer, an interesting phenomenon happens, where changing from dimension $d_i$
to $d_{i+1}$ results in a scaling of the gradient. Since the normal gradient operator happens in euclidean space, a
vector pointing in the direction $[1, 1]$ is "smaller" than a vector pointing in $[1, 1, 1]$. This means that
gradients can quickly scale away from the natural geometry of the loss landscape. The 'trust' region around a weight
update is assumed to be the same in every direction, but this is not reflected in how the gradients are calculated! What
we need is a way to help rescale these gradients so they are essentially comparable as we change dimensionality.

## Operator Norms

One useful change is to instead use the RMS norm to measure lengths in these intemediate layers that change
dimensionality, $|| v ||_{RMS} = \sqrt{1/d} || v ||$. This ensures that we scale by the dimension so that the vectors
are evaluated in the same "units". 

The next step is to try to understand how much the linear layer changes the length of our vector:

$$
|| W ||_{RMS \rightarrow RMS} = \max_{x\neq 0} \frac{||Wx||_{RMS}}{||x||_{RMS}} = \frac{\sqrt{1/d_{out}} ||Wx||_2}{\sqrt{1/d_{in}} ||x||_2} = \sqrt{\frac{f_{in}}{f_{out}}} * ||W||_*
$$


This essentially boils down to a constant relating the change in dimension and the largest singular value of the matrix
in the linear layer. So to control the singular value part of the norm, we can do the SVD of the matrix: $W =
U \Sigma V^T$. Removing $\Sigma$ altogether gets rid of the scaling and leaves us with the principle directions that
the gradient will lie in!

## Wrapping up

There is definitely more to the optimizer, how do we efficiently compute the SVD? What is a Newton-Shulz step
(essentially computes $sign(W)$)? The specifics will probably be good for another dive later on, but essentially there
are some interesting methods that involve specific polynomials that approximate matrix functions that we may want to
compute. In this case we want a polynomial that cheaply computes a Newton-Shulz iteration to get us that orthogonalized
representation of our gradient update. This leaves us with a rough sketch of the muon optimizer update equation!
