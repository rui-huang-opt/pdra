# Private Distributed Resource Allocation Without Constraint Violations

This project is for the numerical verification to the algorithm proposed in Private Distributed Resource Allocation Without Constraint Violations.

## Aim

The proposed algorithm is aimed to solve the multi-agent resource allocation problems with the following setup

$$
\begin{align}
&\min_{\boldsymbol{x}_1,\ldots,\boldsymbol{x}_N}\sum_{i\in\mathcal{I}}f_i(\boldsymbol{x}_i)\\
&s.t.\left\{
\begin{array}{cl}
\sum_{i\in\mathcal{I}}{\boldsymbol{a}_i^1}^\top\boldsymbol{x}_i\leq b_1\\
\vdots\\
\sum_{i\in\mathcal{I}}{\boldsymbol{a}_i^m}^\top\boldsymbol{x}_i\leq b_m\\
\vdots\\
\sum_{i\in\mathcal{I}}{\boldsymbol{a}_i^M}^\top\boldsymbol{x}_i\leq b_M,
\end{array}
\right.
\end{align}
$$

where $\mathcal{I}\coloneqq \{1,2,\ldots,N\}$ representing the indices of the nodes, $\mathcal{M}\coloneqq\{1,2,\ldots,M\}$ representing the indices of the constraints，$f_i(\boldsymbol{x}_i)$ is convex and $\boldsymbol{a}_i^1,\ldots,\boldsymbol{a}_i^M$ hold positive linear independence.

