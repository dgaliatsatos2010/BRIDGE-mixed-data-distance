\# BRIDGE



\*\*Bayesian Redundancy-Aware Information Distance with Graph Ensembles for Mixed-Type Data\*\*



BRIDGE is a methodological framework for constructing redundancy-aware distances for mixed-type data while accounting for conditional predictive novelty and uncertainty in graph structure.



The framework separates mixed-variable commensurability from redundancy and propagates representation and graph uncertainty through a generalized Bayesian (Gibbs) ensemble of directed acyclic graphs.



\## Main features



\- Supports mixed numerical and categorical data.

\- Explicitly addresses redundant and near-redundant variables.

\- Separates marginal similarity from conditional predictive novelty.

\- Incorporates graph-structure uncertainty.

\- Uses generalized Bayesian / Gibbs weighting over DAG ensembles.

\- Supports simulation-based validation and real-data evaluation.

\- Designed for reproducible methodological research.



\## Installation



\### From PyPI



After the first public release, BRIDGE will be installable with:



```bash

pip install bridge-mixed-data-distance

```



\### From source



Clone the repository and install it locally:



```bash

git clone https://github.com/USERNAME/BRIDGE-mixed-data-distance.git

cd BRIDGE-mixed-data-distance

pip install .

```



The `dgaliatsatos2010` placeholder will be replaced with the final GitHub repository URL before release.



\## Basic Python usage



The public Python API will be available through:



```python

import bridge\_mixed\_data\_distance

```



Additional usage examples will be provided in the `examples/` directory.



\## Repository structure



```text

BRIDGE-mixed-data-distance/

├── src/

│   └── bridge\_mixed\_data\_distance/

│       └── \_\_init\_\_.py

├── examples/

├── tests/

├── README.md

├── LICENSE

├── CITATION.cff

├── pyproject.toml

└── .gitignore

```



Additional reproducibility scripts, simulation experiments, and validation materials are included in the research release.



\## Methodological scope



BRIDGE was designed to study mixed-type distance construction under several forms of redundancy, including:



\- exact linear redundancy,

\- near-linear redundancy,

\- nonlinear redundancy,

\- cross-type redundancy,

\- interaction-based redundancy such as XOR structures.



The methodology also evaluates uncertainty arising from alternative graphical representations rather than relying on a single estimated graph.



\## Important interpretation



BRIDGE is not presented as the first redundancy-aware distance for mixed-type data.



The contribution is the integration of:



1\. mixed-type commensurability,

2\. conditional predictive novelty,

3\. redundancy-aware distance construction, and

4\. generalized Bayesian graph-ensemble uncertainty.



The graph weights used by BRIDGE should be interpreted as generalized Bayesian / Gibbs weights over candidate DAGs rather than as a conventional likelihood-based Bayesian-network posterior for the complete joint distribution.



\## Reproducibility



The research repository contains materials for:



\- Monte Carlo simulation experiments,

\- graph-ensemble analyses,

\- real-data validation,

\- ablation studies,

\- sensitivity analysis,

\- runtime evaluation,

\- reproducibility auditing.



Random seeds and saved result files are used where appropriate to facilitate independent verification of reported findings.



\## Citation



If you use BRIDGE in academic work, please cite the software release and the associated methodological paper.



Citation metadata are provided in:



```text

CITATION.cff

```



A permanent Zenodo DOI will be added after the first public GitHub release is archived.



\## License



BRIDGE is released under the MIT License.



\## Author



\*\*Dimitrios Galiatsatos\*\*



Hellenic Open University, Patras, Greece



ORCID: 0009-0005-5232-1425



Email: galiatsatos.dimitrios@ac.eap.gr



\## Version



Current development release:



```text

1.0.0

```

