# Chem_Expl_Comp
Physical Chemistry Expl of Digitizing Ethanol-Cyclohexane Binary Phase Diagram

# Manual
`visualize.py`: generate the interactive phase diagram interface.

`InputTable.py`: interactive interface to input experiment data and compared with standard ethanol-cyclohexane phase diagram.

`train.py`: training ML models.

`importance_anal.py`: SHAP analyst of ML predicetions.

# References
The training data `dataset.xlsx` and `dataset_train.xlsx` was obtained from `Guanlun Sun et al, Chemical Engineering Science, 2023, 282, 119358.`

Other data was obtained from `Gabrielson, S. W. J. Med. Libr. Assoc., 2018, 106, 588.`

Program dependency library: 
 - NumPy 1.26.4: https://numpy.org
 - PyTorch 2.3.1: https://pytorch.org
 - scikit-learn 1.4.2: https://scikit-learn.org/stable/index.html
 - SHAP 0.42.1: https://shap.readthedocs.io/en/latest
 - matplotlib 3.8.4: https://matplotlib.org
 - plotly 5.4.0: https://plotly.com/python
