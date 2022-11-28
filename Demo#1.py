import deepangle as da
# Name of the hdf5 file containing the volumetric image (h5 filed name: /Map)
Data='Data/Data1.h5';
# Extracting contact angles and corresponding points 
Angles,Coordinates=da.getangle(Data)
# Plotting the angle distribution
da.hist(Angles)
# Exporting the results as a text file
da.saveresults(Angles,Coordinates,'Output1')
