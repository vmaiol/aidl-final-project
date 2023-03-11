import netCDF4 as nc
import numpy as np
import torch
dir = './dataNC/p2_2000_182.nc'
ds = nc.Dataset(dir)

print("\n")
print("---------------------------------------------------------------------  \n")
print("METADATA:")
print(ds)
print("---------------------------------------------------------------------  \n")

#Metadata can also be accessed as a Python dictionary, which (in my opinion) is more useful.
#print(ds.__dict__)
#print("--------- \n")

#Then any metadata item can be accessed with its key. For example:
#print(ds.__dict__['start_year'])
#print("--------- \n")

print("\n---------------------------------------------------------------------")
print("DIMENSIONS:")
for dim in ds.dimensions.values():
    print(dim)
print("-----------------------------------------------------------------------\n")

print("\n---------------------------------------------------------------------")
print("VARIABLES VALUES:")
for var in ds.variables.values():
    print(var)
print("-----------------------------------------------------------------------\n")

print("\n---------------------------------------------------------------------")
print("VARIABLES Keys:")
print(ds.variables.keys())
print("\nVariables variables:")
print(ds.variables['variable'][:])
print(ds['input'][0][:].data) #precipitation
exit()
input = ds.variables['input']
print("\nInput:")
#print(input[:])
print("-----------------------------------------------------------------------\n")

# converting list to array
np_array_input = np.array(input)
input_tensor = torch.tensor(np_array_input) # tuple converted to pytorch tensor
print("Tensor:", input_tensor.shape)
print(input_tensor[0])
