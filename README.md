# SUB1D
one-dimensional subsidence model modified from Lees et al 2022

---
## Set-up
```
conda create -f environment.yml
```

# Run
To run SUB1D, you need to create a parameter input file. a couple example input files are provided for reference.
You also need to provide groundwater level time series in csv format for each aquifer layer. The csv files should have two columns
Column 1 -> Date
Column 2 -> Measurement

The csv file names are referenced in the parameter input file.

```
python model.py
```