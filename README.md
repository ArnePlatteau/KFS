# Introduction

This package provides a class for state space models. Upon specification of a state space model, it is possible to
use the kalman filter and kalman smoother. The package provides support for estimating the parameters by maximum likelihood.
The methods are based on the methodology described in Durbin & Koopman (2012).

# Advantages of the package
Several packages for state space methods exist, including the state space methods of statsmodels 
(available under https://www.statsmodels.org/stable/statespace.html#low-level-state-space-representation-and-kalman-filtering). 
However, all packages known to the author provide limited flexibility with regard to specification and estimation of state space models. 
This package aims at providing a very flexible state space implementation. 

More specifically, this implementation allows users to specify their state matrices as any combination of static and time-varying matrices, as long as
the dimensions are consistent. The user can also decide which elements of the state matrices can be considered fixed, and which ones should be 
estimated. The only restriction is that estimated parameters are fixed in time, as to prevent the state space model from overfitting.

# Files 
* kalman.py : definition of the state-spacer class, which contains specification, Kalman filtering, Kalman smoothing, and estimation methods.
* test.py : illustration of using the methods by applying the methods on a dataset.
* nile.dat : dataset containing the data used for the test file

# Future developments
* Extended Kalman filter (to deal with non-linear state space models)
* Collapsed Kalman filter (to deal with high-dimensional models)
* Particle filter (to deal with non-Gaussian, non-linear models)

# Acknowledgements
We use the Nile data from Durbin & Koopman (2012) to illustrate the functioning of the methods.

References
Durbin, James, Koopman Siem-Jan, 2012 Time Series Analysis by State Space Methods.

Copyright (c) 2021 Arne Platteau

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
