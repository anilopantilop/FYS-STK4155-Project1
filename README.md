# FYS-STK4155-Project1
Evaluate 3D data with three different regression methods: OLS, Ridge regression and lasso. Analyse the fit of the model with metrics such as MSE and R2-score. 

## Code examples
The Analysis folder contains examples of how some of the functions are used. Running olsAnalysis.py, RidgeAnalysis.py and LassoAnalysis.py will produce tables and figures used in the report (with some randomization). 

## Real life data
The RealDataAnalysis folder contains the .tif-file with Digital Terrain data used in the report and the python file RealDataAnalysis.py which contains a OLS-example of how to estimate the digital terrain data with bootstrap for error metrics calculation. With minor changes in this file (specifically variations of degree d and changing the method used to produce the model), one can more or less reproduce the real data analysis of our report. 
