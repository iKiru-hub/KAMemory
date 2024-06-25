

## Architecture
---

![[architecture.excalidraw.png]]

Initially, all connections are random. Further, $W_{\text{EC}_{\text{in}}\to \text{CA3}}$ and $W_{\text{EC}_{\text{out}}}$ remain frozen for the entire time.

## Objective
---
During the entire 'lifetime' several stimuli $x$ are experienced and learnt in the CA3$-$CA1 synapses. The goal is to maintain a low decoder error $\mathcal{L}(x,y)$, where $x,y$ are the activity vectors of $\text{EC}_{\text{in}},\text{EC}_{\text{out}}$ respectively.

![[error_plot.excalidraw.png]]

## Training procedure
---
The training procedure is structured in two parts.

**Pre-training** 
The connections $W_{\text{EC}_\text{in}\to\text{CA1}}$ are trained through back-propagation such that the layers $\text{EC}_{\text{in}}-\text{CA1}-\text{EC}_{\text{out}}$ form an auto-encoder.  After this phase, these connections are frozen.

**Lifetime training**
The connections  $W_{\text{CA3-CA1}}$ are learnt through BTSP, implemented in a immediate one-shot manner (*i.e.* no time delay and trace decay). A given input is $x$ from $\text{EC}_{\text{in}}$ is propagated through CA3 to CA1: 
$$
\begin{align}
x_{\text{CA3}} &= W_{\text{EC}_{\text{in}}\to\text{CA3}} \cdot x\\ 
x_{\text{CA1}} &= W_{\text{CA3}\to\text{CA1}} \cdot x_{\text{CA3}}\\
&= W_{\text{EC}_{\text{in}}\to\text{CA3}} \cdot \left(W_{\text{CA3}\to\text{CA1}} \cdot x\right)
\end{align} 
$$ 
an instructive signal is computed in CA1:  $$ IS = W_{\text{EC}_{\text{in}}\to\text{CA1}}\cdot x $$
the weight update quantity of the CA3$-$CA1 layer is calculated as the outer produce of the first and last signal (assumed both to be column vectors): $$ \Delta W_{\text{CA3}\to\text{CA1}} = x_{\text{CA3}} \cdot IS^{T} $$ 
the actual update is then implemented through a learning rate $\eta$ : $$ W_{\text{CA3}\to\text{CA1}}\leftarrow \eta\,W_{\text{CA3}\to\text{CA1}} $$ 
then, the final output is defined as: 
$$
y=W_{\text{CA1}\to\text{EC}_\text{out}}\cdot x_{\text{CA1}}
$$ 

**new $IS$**
In order to prevent unbounded grow of the synapses of re-experienced stimuli, the instructive signal can be defined to quantify the dissimilarity of the feedforward input $x$ and the retrieved memory $x_{\text{CA3}}$:
$$
IS = 1 - \cos\left(W_{\text{EC}_{\text{in}}\to{\text{CA1}}}\cdot x, x_{\text{CA3}}\right)
$$
with this tweak, the hope is that the reached the [objective](#Objective).

