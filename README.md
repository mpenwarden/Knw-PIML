# Kolmogorov n-widths for Multitask Physics-informed Machine Learning (PIML) Methods

## Abstract

Physics-informed machine learning (PIML) as a means of solving partial differential equations (PDE) has
garnered much attention in the Computational Science and Engineering (CS\&E) world. This topic encompasses a broad array of methods and models aimed at solving single or multitask PDE problems. PIML is identified by the incorporation of physical laws into the training process of machine learning models in lieu of large data when solving PDE problems. Despite the overall success of this collection of methods, it remains incredibly hard to analyze, benchmark, and generally compare one approach to another. Using Kolmogorov n-widths as a measure of effectiveness of approximating functions, we judiciously apply this metric in the comparison of various multitask PIML architectures. We compute lower accuracy bounds of certain models with which benchmarks can be defined on a variety of PDE problems. We also incorporate this metric into the optimization process, which improves the models' generalizability over the multitask PDE problem. Finally, we provide theoretical results of the convergence of approximating the Kolmogorov n-width for PIML models. 

## Citation

TBD

## Description of codebase

TBD

## Examples

### Example: 1D Poisson

MH-PINN (sin) Kolmogorov n-width = $1.7 \times 10^{-4}
https://github.com/mpenwarden/Knw-PIML/assets/74904442/ce1ed207-9e5a-4b22-a48d-5937cf8fe66e

MH-PINN (tanh) Kolmogorov n-width = $7.5 \times 10^{-2}
https://github.com/mpenwarden/Knw-PIML/assets/74904442/c863cec0-965e-4187-ba21-3b2cc02dc022


## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.
