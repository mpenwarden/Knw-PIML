# Kolmogorov n-Widths for Multitask Physics-Informed Machine Learning (PIML) Methods: Towards Robust Metrics

## Abstract

Physics-informed machine learning (PIML) as a means of solving partial differential equations (PDE) has
garnered much attention in the Computational Science and Engineering (CS&E) world. This topic encompasses a broad array of methods and models aimed at solving a single or a collection of PDE problems, called
multitask learning. PIML is characterized by the incorporation of physical laws into the training process of
machine learning models in lieu of large data when solving PDE problems. Despite the overall success of
this collection of methods, it remains incredibly difficult to analyze, benchmark, and generally compare one
approach to another. Using Kolmogorov n-widths as a measure of effectiveness of approximating functions,
we judiciously apply this metric in the comparison of various multitask PIML architectures. We compute
lower accuracy bounds and analyze the model’s learned basis functions on various PDE problems. This
is the first objective metric for comparing multitask PIML architectures and helps remove uncertainty in
model validation from selective sampling and overfitting. We also identify avenues of improvement for model
architectures, such as the choice of activation function, which can drastically affect model generalization to
“worst-case” scenarios, which is not observed when reporting task-specific errors. We also incorporate this
metric into the optimization process through regularization, which improves the models’ generalizability
over the multitask PDE problem.

## Citation

Penwarden, Michael, Houman Owhadi, and Robert M. Kirby. "Kolmogorov n-Widths for Multitask Physics-Informed Machine Learning (PIML) Methods: Towards Robust Metrics." arXiv preprint arXiv:2402.11126 (2024).

https://arxiv.org/abs/2402.11126

## Description of codebase

This repository archives the results constituting the numerical experiments section of the paper linked above. The repository is split into three parts:

Data - The data folder contains the reference PDE solutions to the problems solved in the manuscript.

Results - The results folder contains MH-PINN and PI-DON model runs that constitute the results (1D Poisson & 2D nonlinear Allen-Cahn) reported in the manuscript.

Source - The source folder contains the final version of the foundational classes and helper functions, which are called in the Jupyter Notebook files in the Results folder to solve the PDE problems described.

## Example: 1D Poisson (Animations of the competitive optimization used in computing the Kolmogorov n-Width)

$\frac{\partial^2 u}{\partial x^2} = f(x)$ <br>
$x \in [-1,1]$ <br>
$u(-1) = u(1) = 0$

MH-PINN (sine activation function) Kolmogorov n-width (reported as Rel. $L_2$ Error) = $1.9 \times 10^{-2}$

https://github.com/mpenwarden/Knw-PIML/assets/74904442/819a1146-c489-4616-87cb-063ce1f64225

MH-PINN (tanh activation function) Kolmogorov n-width (reported as Rel. $L_2$ Error) = $30.1 \times 10^{-2}$

https://github.com/mpenwarden/Knw-PIML/assets/74904442/7e71d694-da4a-4ec1-9173-e2a0808399dc


## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.
