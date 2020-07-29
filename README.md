# Entropy estimators benchmark

Estimators of Shannon entropy and mutual information for random variables. Bivariate and multivariate. Discrete and continuous.

Estimators:
    
* [NPEET](https://github.com/gregversteeg/NPEET): Non-parametric Entropy Estimation Toolbox
* [GCMI](https://github.com/robince/gcmi): Gaussian-Copula Mutual Information
* [MINE](https://arxiv.org/pdf/1801.04062.pdf): Mutual Information Neural Estimation
* [IDTxl](https://github.com/pwollstadt/IDTxl): Information Dynamics Toolkit xl

For different distribution families, one test is performed to compare estimated entropy or mutual information with the true (theoretical) value.

The input to estimators are two-dimensional arrays of size `(len_x, dim_x)`.

### Results

For complete analyses, refer to [Entropy Estimation](results/entropy) and [Mutual Information Estimation](results/mutual_information) results.

To illustrate an example, below is a benchmark of mutual information estimation of normally distributed X and Y covariates:

![](results/mutual_information/images/distributions/_mi_normal_correlated.png)


### Running locally

To run tests locally,

1. Clone all the submodules:

   `git clone --recurse-submodules https://github.com/dizcza/entropy-estimators.git`

2. Install the requirements:

   ```
   conda env create -f environment.yml
   conda activate entropy-estimators
   ```

3. Run the benchmark:

    * entropy: `python benchmark/entropy_test.py`
    * mutual information:
      - distributions: `python benchmark/mutual_information/distributions.py`
      - classifier: `python benchmark/mutual_information/classifier.py`
