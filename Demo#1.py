import deepangle as da
Data='Data/Data1.h5';
Angles,Coordinates=da.getangle(Data,Mode='contact',Fast=1)
da.hist(Angles)
da.saveresults(Angles,Coordinates,'Output1')
