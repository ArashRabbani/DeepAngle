import deepangle as da
import scipy.io as sio
# Reading a Matlab saved variable from the data directory that contains a variable called "A"
Data=sio.loadmat('Data/Data2.mat')  
Data=Data['A']
# Visulizing 3 mid-slices of the volumetric image
da.sliceshow(Data)
# Extracting contact angles and corresponding points by explicitely giving the numpy array as input
Angles,Coordinates=da.getangle(Data)
# Plotting the angle distribution
da.hist(Angles)
# Exporting the results as a text file
da.saveresults(Angles,Coordinates,'Output1')