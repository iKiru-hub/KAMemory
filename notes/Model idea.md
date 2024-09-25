## Architecture
---

![[architecture.excalidraw|700]]

**Auto-encoder**
Identified in $\text{EC}_{\text{in}}\to \text{CA1}\to \text{EC}_{\text{out}}$

**MTL**
Identified in $\text{EC}_{\text{in}}\to \text{CA3}\to \text{CA1}\to \text{EC}_{\text{out}}$ further, $W_{\text{CA3}\to \text{CA1}}$ are initially zero.

## Objective
---
During the entire 'lifetime' several stimuli $x$ are experienced and learnt in the CA3$-$CA1 synapses. The goal is to maintain a low decoder error $\mathcal{L}(x,y)$, where $x,y$ are the activity vectors of $\text{EC}_{\text{in}},\text{EC}_{\text{out}}$ respectively.

**Decodability during learning**
- downstream regions should maintain their interpretation of HP representations during learning


![[error_plot.excalidraw.png]]


**Stimuli**
As for now, the stimuli are patterns of size $N_{\text{EC}_{\text{in}}}$ with $K$ non-zero entries (ones) sampled from a uniform distribution.

**Activation function**
It is used a new activation function $\sigma_{\beta, k}$ , we called it *sparsemoid*:
$$
\begin{align}
\theta&=\frac{\text{sorted}(z)_{k} +\text{sorted}(z)_{k+1}}{2}\\\\
z &= \sigma_{\beta,\theta}(z) \\
\end{align}
$$
basically, it sets the input vector $z$ to have around $k$ active neurons with a value in $(0, 1)$ by defining a variable threshold $\theta$ and using it in a generalized sigmoid $(1 + \exp{(-\beta\,(x - \theta)}))^{-1}$ 

This activation function has thus two hyper-parameters $\{\beta, k\}$ which can be tailored for specific layers. 

## Training procedure
---
The training procedure is structured in two parts.


**Pre-training** 
The connections $W_{\text{EC}_\text{in}\to\text{CA1}}$ are trained through back-propagation such that the layers $\text{EC}_{\text{in}}-\text{CA1}-\text{EC}_{\text{out}}$ form an auto-encoder.  After this phase, these connections are frozen.


**Lifetime training**
The connections  $W_{\text{CA3-CA1}}$ are learnt through BTSP, implemented in a immediate one-shot manner (*i.e.* no time delay nor trace decay).
A given input is $x$ from $\text{EC}_{\text{in}}$ is propagated through CA3 to CA1 by applying the dot product $\cdot$ and the *sparsemoid*.
$$
\begin{align}
x_{\text{CA3}} &= \sigma(W_{\text{EC}_{\text{in}}\to\text{CA3}} \cdot x)\\ 
x_{\text{CA1}} &= \sigma(W_{\text{CA3}\to\text{CA1}} \cdot x_{\text{CA3}})\\
&=\sigma(W_{\text{CA3}\to\text{CA1}} \cdot\left(W_{\text{EC}_{\text{in}}\to\text{CA3}}\cdot x\right))
\end{align} 
$$ 
an instructive signal is computed in CA1:  $$ IS = \sigma(W_{\text{EC}_{\text{in}}\to\text{CA1}}\cdot x)$$
the weight update quantity of the CA3$-$CA1 layer is calculated as the outer product of the first and last signal (assumed both to be column vectors): $$ \Delta W_{\text{CA3}\to\text{CA1}} = x_{\text{CA3}} \cdot IS^{T} $$ 
Usually, the update is implemented through a learning rate $\eta$ : 
$$ 
W_{\text{CA3}\to\text{CA1}}\leftarrow (1-\eta)W_{\text{CA3}\to\text{CA1}} +\eta\,\Delta W_{\text{CA}\to\text{CA1}} 
$$
here, we take a similar approach with $\eta=IS$ but re-defining the weight update as the CA3 activity  $\Delta W=x_{\text{CA3}}$ . A possible rationale is that the instructive signal *selects* the CA3$\to$CA1 synapses to overwrite with a new value, corresponding to the CA3 neural activity of the new pattern: 
$$ 
\begin{align}
W_{\text{CA3}\to\text{CA1}}&\leftarrow (1-IS)\odot W_{\text{CA3}\to\text{CA1}} +x_{\text{CA3}}\cdot IS^{T}
\end{align}
$$

Then, the final output is then calculated as: 
$$
y=\sigma(W_{\text{CA1}\to\text{EC}_\text{out}}\cdot x_{\text{CA1}})
$$ 

## Results 
---
![[reconstruction_1.png|690]]
*first plot* : patterns (rows) from 0 to 4
*second plot* : reconstruction from the autoencoder
*third plot* : reconstruction from the MTL (through CA3)


#### Memory capacity
---

code snippet
```python

# outputs.shape : (#patterns, #patterns)
# parameters : idx_pattern, nsmooth, threshold

# select the first pattern and pad it
padded_out = np.pad(outputs[idx_pattern:, idx_pattern],
					(nsmooth-1, 0), mode="edge")

# smooth it
outputs = np.convolve(padded_out,
				  np.ones(nsmooth)/nsmooth,
				  mode="valid")

# find the highest index where the output
# is below the threshold
idx = np.argmin(np.where(outputs >= threshold,
						 outputs, -np.inf),
				axis=0).item()

```


- [x] define a good threshold $\theta$ for the memory capacity -> hyper-parameter
- [x] enquiry the problem with the below-threshold calculations


![[accuracy_0.png]]
***Recollection accuracy for all patterns**

![[capacity_0.png]]
****Recollection accuracy for a given pattern over the the number patterns stored afterwards, for multiple values of alpha***

![[capacity_over_alpha.png|670]]
***Recollection accuracy for all patterns for multiple values of alpha***


![[rcapacities_193733.gif]]



## Grid search
---
- [x] grid search over the parameters with the capacity metric found above -> good model

- sample random points from a uniform.

1) optimize the network (also bayesian is fine)
2) understanding how capacity depends on the parameters (full)

**Weights & Biases**
https://wandb.ai/ikiru-university-of-oslo/kam_2/sweeps/tc96txc8?nw=nwuserikiru


## Plans
---

#### Goals
- [ ] send abstract to Claudia <--- 14.10
- [ ] submit to Cosyne (end of October)
- [ ] write paper <--- 2024

#### proximal TODO

**Albert**
- [ ] architecture figure
- [ ] btsp figure
- [ ] ecological environment

**Kiru**
- [ ] training figure <--- *28.09*
- [ ] matrix figure   <--- *28.09*
- [ ] pc visualization | task
- [ ] prediction

