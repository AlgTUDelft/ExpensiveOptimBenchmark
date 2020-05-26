# ExpensiveOptimBenchmark

<!-- TODO: Better name! -->

<!-- TODO: Insert general idea of why this benchmark is needed here -->

## Usage

Running this benchmark requires a working installation of python 3. 

- A virtualenv can be created through the use of [Poetry](https://github.com/python-poetry/poetry), which will automatically incorporate the neccesary dependencies:
    ```
    poetry install
    ```
- Once finished, you can open a shell in the virtualenv by running
    ```
    poetry shell
    ```
    or run a singular command using
    ```
    poetry run python ...
    ```
<!-- TODO: Once added, add method for running approach here -->

## References

### Problems
- None yet!
<!-- TODO: Add benchmark problems -->

#### Possible additions:
- ML Hyperparameter Optimization / AutoML

### Reference Algorithms
- pyGPGO: https://github.com/josejimenezluna/pyGPGO
    > Jiménez, J., & Ginebra, J. (2017). pyGPGO: Bayesian Optimization for Python. The Journal of Open Source Software, 2, 431.
- HyperOpt: https://github.com/hyperopt/hyperopt
    >  Bergstra, J., Yamins, D., Cox, D. D. (2013) Making a Science of Model Search: Hyperparameter Optimization in Hundreds of Dimensions for Vision Architectures. To appear in Proc. of the 30th International Conference on Machine Learning (ICML 2013).
- SMAC: https://github.com/automl/SMAC3
    > Hutter, F. and Hoos, H. H. and Leyton-Brown, K. Sequential Model-Based Optimization for General Algorithm Configuration In: Proceedings of the conference on Learning and Intelligent OptimizatioN (LION 5)

#### Possible additions:
- scikit-learn: https://github.com/scikit-learn/scikit-learn
    > @article{scikit-learn,
    >  title={Scikit-learn: Machine Learning in {P}ython},
    >  author={Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V.
    >          and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.
    >          and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and
    >          Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.},
    >  journal={Journal of Machine Learning Research},
    >  volume={12},
    >  pages={2825--2830},
    >  year={2011}
    > }
- MOE: https://github.com/Yelp/MOE
- Spearmint: https://github.com/HIPS/Spearmint
    > Practical Bayesian Optimization of Machine Learning Algorithms. Jasper Snoek, Hugo Larochelle and Ryan Prescott Adams. In Advances in Neural Information Processing Systems, 2012  

    > Multi-Task Bayesian Optimization. Kevin Swersky, Jasper Snoek and Ryan Prescott Adams. In Advances in Neural Information Processing Systems, 2013  

    > Input Warping for Bayesian Optimization of Non-stationary Functions. Jasper Snoek, Kevin Swersky, Richard Zemel and Ryan Prescott Adams. In International Conference on Machine Learning, 2014  

    > Bayesian Optimization and Semiparametric Models with Applications to Assistive Technology. Jasper Snoek, PhD Thesis, University of Toronto, 2013  

    > Bayesian Optimization with Unknown Constraints. Michael Gelbart, Jasper Snoek and Ryan Prescott Adams. In Uncertainty in Artificial Intelligence, 2014