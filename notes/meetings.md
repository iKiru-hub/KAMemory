
**Meeting 20.8.24**

weights EC $\to$ CA3 as an invertible matrix

capacity:
- pattern length 
- number of patterns

Roadmap:
- *week 1*: simulation results
- *week 2*: experimental relevance (fit)
- *week 3*: experimental predictions

**notes**
- plot the distribution of activations in $CA1$ and $EC_{out}$ to check that the average norm increases with low values of $\alpha$


**Slides**
1. summary of BTSP, papers and overview
2. question: *decodability problem*
3. describe the model and the choices in light of a minimalist drive
4. explain figure (4) and how our rule is a simplified version (it approximates the original rule) as a square function, with LTP and LTD acting immediately with the pairing IS-ET
5. explain the architecture and its online/offline dynamics, areas and training
6. describe the simulation setup + nice figures
7. topic: how to merge it with experimental evidence? what predictions can be made?



**meeting** 24.9.24
![[Pasted image 20240924143040.png]]
- ecological task (no sp/nsp distinction)
- random IS: same shape but random entries (i.e. as if it was shuffled) | bc BTSP meant to be random somehow



**Meeting** 8.10.24

- fewer patterns
- deadline for 10

Linear track:
- simpler
- experiments are mostly using it
- sensory cue spread over the environment

Autoencoder:
- mixed inputs (spatial + non-spatial)
- obtain spatial cells in the latent layer


![[Pasted image 20241008152511.png]]


**Meeting** 10.10.24

linear track
- group of 2 laps
- use of ca1 activity and position to evaluate the tuning wrt space

1. how to generate the input
	- two random sensory cues
	- iterate over laps and assign
	- record the info for which lap contains which cue

2. run the agent
3. split CA1 activity wrt lap containing a specific cue
4. mean activity of CA1 neurons over laps for a specific cue

plot idea:
- show the pc tuning in both cases

**Meeting** 12.10.24

- [x] *sensitivity of IS*
- goal: find neurons in IS (latent layer of AE) strongly tuned to the cue(s):
	- plot 1 : distribution of sensitivity
	- plot 2 heatmap: 1st group selective for cue 1, 2nd group for 2nd
- how: run laps
	- record the sensitivity of IS when cue_x is present wrt to the average activity (laps for cue1). i.e. sensitivity as activity/average_activity

- [x] *CA1 tuning*
- use the distribution of IS to sort CA1 activity and plot the resulting distribution
- goal: see the influence to CA1 tuning

idea:
- place fields are place + cue fields


**Meeting** 13.10.25

@albert
**abstract**

@kiru
**plots**
- [ ] prettify
- [x] clean the notebook


**Meeting** 14.10.25
@kiru
- [x] decodability figure: already with place and sensory input pattern
slight difference in MTL performance in:
```python
model.load_state_dict(
	# torch.load(f"{cache_dir}/{session}/autoencoder.pt"))
	torch.load(f"{cache_dir}/{session}/autoencoder.pt",
			   weights_only=True))
```
+binarize option of the spatial input

- [x] try same position $\neq$ cue
plot difference between IS-indexed (normal) and CA1-indexed (bad)


accuracy plot with lec data is weird



---
colorbars
*figure d*
- initial weight
xlabels: 
- ECin
- ECout (IS for ECin)
- ECout (IS random)

figure f:
- axylines for separating the neuronal groups
- axvlines same color as in figure c

figure e:
- ca1 tuning merged 

figure c:
- add labels 'space', 'sensory', ...



---
##### MORE TODO

@kiru
- [ ] **author list**

--- 
first figure: **introduction to the question - model**
- a) problem diagram ~neuroscience
- b) architecture
- c)![[error_plot.excalidraw.png]]
- d) results with random IS

second figure: **decodability results** 
- a) results with our IS
- b) ![[error_plot.excalidraw.png]] w/ results
- c)

third reproducibility:
- a) experimental layout
- b) over-repr.
- c) ca1 sorting
- d) memory matrix | -> recency | are there experiments?
- e) pc moving in space

bonus
- f) sparsity to tuning

fourth
- decomposition of the memory trace
- predictions


---
experiments inspired by
- btsp curve

- changing the cue position

- mild code refactoring



---
the more a neuron is spatially tuned then its spatial remapping is more pronounced