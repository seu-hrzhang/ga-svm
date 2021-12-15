# Genetic Algorithm Optimized Support Vector Machine

## Introduction

This project is the Python implementation of **Support Vector Machine** (SVM), which is solved using conventional **Platt SMO** and **Genetic Algorithm** (GA), respectively.

The SVM in this project is used to solve a 2-dimensional linear classification problem. Source data to be classified was generated randomly, as shown below:

<img title="" src="https://gitee.com/seu-hrzhang/ga-svm/raw/master/figure/input-data.png" alt="" data-align="center" width='400'>

In this project, we assume the input data to be linear-separable, otherwise the program may encounter unexpected problems.

## Theory

### Linear Support Vector Machine (SVM)

Let the input data be $m$ samples $(x_1,y_1),\dots,(x_m,y_m)$, where $x\in\mathbb{R}^n$ is the feature vector and $y\in\mathbb{R}$ is the binary output, taking value from $\{1,-1\}$.

The standard form of SVM is expressed as

$$
\begin{aligned}
& \substack{\max\\\alpha} && \frac{1}{2}\sum_{i=1}^{m}\sum_{j=1}^{m}\alpha_i\alpha_jy_iy_j(x_i \cdot x_j)-\sum_{i=1}^m\alpha_i \\
& \text{s.t.} && \sum_{i=1}^{m}\alpha_iy_i=0 \\
& && \alpha_i \ge 0,\ i=1,\dots,m
\end{aligned}
$$

where $\alpha$ is the equivalent optimization vector brought by Lagrange Function. The equation obove uses a so-called hard margin to separate two hyperplanes. To address avoidance of outliers influences, soft margin is introduced, and the second constraint becomes

$$
0 \le \alpha_i \le C,\ i=1,\dots,m
$$

where $C$ is a punishing constant. Therefore the dual complementarity [KKT conditions](https://en.wikipedia.org/wiki/Karush–Kuhn–Tucker_conditions) of the problem can be expressed as

$$
\alpha_i^\star\big[y_i(w^\mathrm{T}x_i+b)-1+\xi_i^\star\big]=0
$$

which can be derivied into

$$
\begin{aligned}
\alpha_i^\star=0 & \Rightarrow y_ig(x_i) \ge 1 \\
0<\alpha_i^\star<C & \Rightarrow y_ig(x_i)=1 \\
\alpha_i^\star=C & \Rightarrow y_ig(x_i) \le 1
\end{aligned}
$$

The aforementioned constraints should be followed during optimization. Further introduction of SVM can be found on [Wikipedia](https://en.wikipedia.org/wiki/Support-vector_machine).

### Sequential Minimal Optimization (SMO)

The problem above involves convex constrained optimization, which is commonly solved using the SMO. The problem consists of $m$ variables, causing vast difficulty in optimization solving. SMO optimizes 2 specific variables at a time and regard the rest as constants. From the first constrain in SVM definition, while we cofirm $\alpha_3,\dots,\alpha_m$, the linear relation between $\alpha_1$ and $\alpha_2$ is acquired as well:

$$
\alpha_1y_1+\alpha_2y_2=-\sum_{i=3}^m\alpha_iy_i=-\zeta
$$

From [Wikipedia](https://en.wikipedia.org/wiki/Sequential_minimal_optimization), the updating formula of $\alpha_2$ is

$$
\alpha_2^{new}=\alpha_2^{old}+\frac{y_2(E_1-E_2)}{K}
$$

where $E_i=f(x_i)-y_i$, $f(x_i)=\sum_{j=1}^{n}y_j\alpha_jK_ij+b$ and $K_{ij}$ refers to the kernel operation of $x_i$ and $x_j$. Then considering the constraints $0 \le \alpha_i \le C$, $(\alpha_i,\alpha_j)$ only possibly falls in the rectangle $[0,C]\times[0,C]$, which should be checked during optimization.

In general, the algorithm proceeds as follows:

1. Find a Lagrange multiplier $\alpha_1$ that violates the KKT conditions for the optimization problem.

2. Pick a second multiplier $\alpha_2$ and optimize the pair $(\alpha_1,\alpha_2)$.

3. Repeat steps 1 and 2 until convergence.

### Genetic Algorithm (GA)

Genetic Algorithm (GA) is a metaheuristic inspired by the process of natural selection that belongs to the larger class of evolutionary algorithms (EA). Genetic algorithms are commonly used to generate high-quality solutions to optimization and search problems by relying on biologically inspired operators such as mutation, crossover and selection.

Using GA to solve SVM is easy as toe. By applying the optimization target as the fitness function, we can easily get the fitness of each individual.

## Usage

1. Using a naive SMO solver (optimization variables selected completely randomly):
   
   ```python
   smo_solver = SMO(data, labels, 0.8, 0.001, max_it=40)
   smo_solver.naive_smo()
   smo_solver.plot('Support Vectors & Hyperplane')
   ```

2. Using a Platt SMO solver:
   
   ```python
   smo_solver = SMO(data, labels, 0.8, 0.001, max_it=40)
   smo_solver.platt_smo()
   smo_solver.plot('Support Vectors & Hyperplane')
   ```

3. Using a GA solver:
   
   ```python
   ga_solver = GA(data, labels, 50, 3, 1000)
   ga_solver.evolve()
   ga_solver.plot('Hyperplane')
   ```

*TODO:* Tesing Section