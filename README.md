
LSTM Network template for regression

Predict real valued output over multiple timesteps, given sequential inputdata. Formally, this LSTM network fits parameters to compute the function:

R^(timesteps, input_dim) --> R^output_dim

Where

timesteps = consecutive timesteps to be considered a single example

input_dim = number of independent variables to be used in regression. Could be eg. data from various sensors

output_dim = output can be either scalar or vector - define dimensionality here

NOTE! This is an exercise work done at deep learning seminar / TUT Pori Campus / Spring 4.5.2018. No guarantees are given whatsoever. Use at your own risk.
