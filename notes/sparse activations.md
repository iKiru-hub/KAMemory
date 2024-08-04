

## Sparsemax
---
**Algorithm description (forward)**
$$
\begin{align*}
\text{Given:} \quad & \mathbf{z} \in \mathbb{R}^K \quad (\text{input scores}) \\
\text{Step 1:} \quad & \text{Sort } \mathbf{z} \text{ in descending order to get } \mathbf{z}_{\text{sorted}} \\
& \mathbf{z}_{\text{sorted}} = (z_{[1]}, z_{[2]}, \ldots, z_{[K]}) \quad \text{where} \quad z_{[1]} \geq z_{[2]} \geq \ldots \geq z_{[K]} \\
\text{Step 2:} \quad & \text{Compute the cumulative sum of } \mathbf{z}_{\text{sorted}} \\
& S_k = \sum_{j=1}^k z_{[j]}, \quad k = 1, 2, \ldots, K \\
\text{Step 3:} \quad & \text{Find the threshold } \tau \text{ that satisfies} \\
& \tau = \frac{S_k - 1}{k} \quad \text{where} \quad k = \max \{ j \in [K] \mid z_{[j]} > \frac{S_j - 1}{j} \} \\
\text{Step 4:} \quad & \text{Compute the output } \mathbf{p} \\
& p_i = \max(z_i - \tau, 0), \quad \forall i = 1, 2, \ldots, K \\
\text{Output:} \quad & \mathbf{p} \quad (\text{sparse probability distribution})
\end{align*}
$$

**Code**
```python
def sparsemax(z: np.ndarray) -> np.ndarray:

    # step 1
    z_sorted = z[np.argsort(-z)]

    # step 2
    col_range = (np.arange(len(z))+1)
    cumsum = np.cumsum(z_sorted)
    lhs = (1 + col_range * z_sorted)

    ks = lhs > cumsum
    k_z = np.max(col_range * ks)

    # step 3
    tau_z = (z_sorted[:k_z].sum()-1)/k_z

    # step 4
    out = np.maximum(z-tau_z, 0)
    return out

```
**Code output example**
```
z:         [1.764 0.4   0.979 2.241 1.868]
z_sorted:  [2.241 1.868 1.764 0.979 0.4  ]
col_range: [1 2 3 4 5]
cumsum:    [2.241 4.108 5.873 6.851 7.251]
lhs:       [3.241 4.735 6.292 4.915 3.001]
ks:        [ True  True  True False False]
k_z:       3
tau_z:     1.624
out:       [0.14  0.    0.    0.617 0.243]
```

**Sources**
https://medium.com/deeplearningmadeeasy/sparsemax-from-paper-to-code-351e9b26647b
ChatGPT


## Softmax + Sigmoid
---

Softmax
$$
\phi_{\gamma}(\mathbf{z})=\frac{\exp(\gamma\,\mathbf{z})}{\sum_{i}\exp(\gamma\,\mathbf{z}_{i})}
$$
Sigmoid
$$
\sigma_{\beta,\alpha}(\mathbf{z})=\frac{1}{1+\exp(-\beta\,(\mathbf{z}-\alpha))}
$$

Final activation as $\varphi_{\gamma, \beta, \alpha}(\mathbf{z})=\sigma_{\beta,\alpha}\,\circ\,\phi_{\gamma}\,(\mathbf{z})$ 

**Backward pass**

Backpropagating through the softmax $\mathbf{a}_{1}=\phi_{\gamma}(\mathbf{z})$ yields a Jacobian matrix:
$$
\frac{\partial\mathbf{a}_{1}}{\partial\mathbf{z}}=\gamma\,\mathbf{a}_{1}(\delta-\mathbf{a}_{1})
$$
while the derivative of the sigmoid $\mathbf{a}_{2}=\sigma_{\beta,\alpha}(\mathbf{a}_{1})$  yields:
$$
\frac{\partial\mathbf{a}_{2}}{\partial\mathbf{a}_{1}}=\beta\,\mathbf{a}_{2}(1 - \mathbf{a}_{2})
$$
Putting it all together:
$$
\begin{align}
\varphi_{\gamma,\beta,\alpha}(\mathbf{z})&=\frac{\partial\mathbf{a}_{2}}{\partial\mathbf{a}_{1}}\cdot\frac{\partial\mathbf{a}_{1}}{\partial\mathbf{z}} \\
&=\beta\,\mathbf{a}_{2}(1-\mathbf{a}_{2})\cdot\gamma\,\mathbf{a}_{1}(\delta-\mathbf{a}_{1}) \\
&=\beta\gamma\cdot \left[\sigma(\phi(\mathbf{z}))\cdot \phi(\mathbf{z})\right] \cdot\left[(1-\sigma(\phi(\mathbf{z})))\cdot (\delta-\phi(\mathbf{z}))\right]
\end{align}
$$

It is also possible to compose a *relu* $\mathbf{a}_{3}=\psi(\mathbf{z})=\max(0,\mathbf{z})$, with derivative:
$$
\begin{align}
\psi^{-1}(\mathbf{z})&=
\begin{cases}
0\quad\text{if } z<0 \\
1\quad\text{otherwise}
\end{cases}\\
&=\mathbb{1}_{\mathbf{z}>0}
\end{align}
$$
yielding:

$$
\begin{align}
\varphi_{\gamma,\beta,\alpha}(\mathbf{z})&=\frac{\partial\mathbf{a}_{3}}{\partial\mathbf{a}_{2}}\cdot\frac{\partial\mathbf{a}_{2}}{\partial\mathbf{a}_{1}}\cdot\frac{\partial\mathbf{a}_{1}}{\partial\mathbf{z}} \\
&=\mathbb{1}_{\sigma(\phi(\mathbf{z}))>0}\cdot\beta\gamma\cdot \left[\sigma(\phi(\mathbf{z}))\cdot \phi(\mathbf{z})\right] \cdot\left[(1-\sigma(\phi(\mathbf{z})))\cdot (\delta-\phi(\mathbf{z}))\right]
\end{align}
$$







