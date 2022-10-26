import scipy.ndimage as ndi
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
import h5py
import copy
import math
import random
from joblib import Parallel, delayed
import joblib
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Input, Conv3D, MaxPooling3D, Dense, Flatten, Reshape, BatchNormalization
import edt # Multithread distance transform https://github.com/seung-lab/euclidean-distance-transform-3d
random.seed(1)
np.random.seed(1)
def shiftmatrix(A,x,*args):
    B=copy.deepcopy(A)
    if len(args)==0:
        N=1 # N shows the shift amount and x shows the shift direction
    else:
        N=args[0]
    if len(A.shape)==3:    
        if x==1:
            B[:-N,:,:]=B[N:,:,:]
        if x==2:
            B[:,:-N,:]=B[:,N:,:]
        if x==3:
            B[:,:,:-N]=B[:,:,N:]
        if x==-1:
            B[N:,:,:]=B[:-N,:,:]
        if x==-2:
            B[:,N:,:]=B[:,:-N,:]
        if x==-3:
            B[:,:,N:]=B[:,:,:-N]
    if len(A.shape)==2:    
        if x==1:
            B[:-N,:]=B[N:,:]
        if x==2:
            B[:,:-N]=B[:,N:]
        if x==-1:
            B[N:,:]=B[:-N,:]
        if x==-2:
            B[:,N:]=B[:,:-N]
    return B
def trim(A,N):
    if A.ndim==3:
        A=A[N:-N,N:-N,N:-N]
    if A.ndim==2:
        A=A[N:-N,N:-N]
    return A
def margin(A,N):
    if A.ndim==3:
        B=np.zeros((np.int(A.shape[0])+N*2,np.int(A.shape[1])+N*2,np.int(A.shape[2])+N*2))
        B[N:-N,N:-N,N:-N]=A
    if A.ndim==2:
        B=np.zeros((np.int(A.shape[0])+N*2,np.int(A.shape[1])+N*2))
        B[N:-N,N:-N]=A
    return B
def sph(r,dim=3):
    if dim==3:
        s=np.int(np.ceil(2*(r)+1));   
        A=np.zeros((s,s,s)); 
        A[np.int((s-1)/2),np.int((s-1)/2),np.int((s-1)/2)]=1;
        r=np.float(r); 
        A=morph(A,r-.45,'dil')
    if dim==2:
        s=np.int(np.ceil(2*(r)+1));   
        A=np.zeros((s,s)); 
        A[np.int((s-1)/2),np.int((s-1)/2)]=1;
        r=np.float(r); 
        A=morph(A,r-.45,'dil')
    return A
def bwdist(A):

    A=A.astype(bool)
    A=np.asfortranarray(A)
    B = edt.edt(A, parallel=4 ,order='F')
    return B
def morph(A,r,method):
    if (method=='dil') | (method=='dilate'):
        return bwdist(1-A)<=(r+.5)
    if (method=='ero') | (method=='erode'):    
        return bwdist(A)>=(r+.5) 
    if (method=='ope') | (method=='open'): 
        return morph(morph(A,r,'ero'),r,'dil')
    if (method=='clo') | (method=='close'): 
        return morph(morph(A,r,'dil'),r,'ero')
def hist(Angles):
    plt.hist(Angles,np.arange(0,180),density=True,histtype=u'step')
    plt.rcParams["font.family"] = "serif"
    plt.xlabel('Angle (degree)')
    plt.ylabel('Relative frequency')
    plt.xlim([0,180])

def corners(image): # gives the coordinate of angles in the black space
    from skimage.feature import corner_harris, corner_peaks
    coords = corner_peaks(corner_harris(image), min_distance=7)
    return coords
def cornerpoints(A,Rad):
    P=np.zeros((1,3))
    SliceList=np.int32(np.linspace(Rad+1,A.shape[2]-1-Rad,10))
    for I in SliceList:
        loc=I
        B=np.squeeze(A[:,:,loc])
        ID=corners(B)    
        if len(ID)==0:
            continue
        ID=np.unique(ID, axis=0)
        yy=np.concatenate((ID,np.tile(loc,(ID.shape[0],1))),axis=1)
        P=np.concatenate((P,yy),axis=0)
    P=np.int32(P)    
    return P
def corner_gen(r,T): # Generates a geometry with T corner angle in degree and 2*r+1 is the size of the geometry 
    A=np.ones((2*r+1,2*r+1,2*r+1))
    A[r:,:,:]=0
    A=ndi.rotate(A,np.asscalar(np.float32(T)),axes=(0,1),mode='nearest',reshape = False,prefilter=False,order=0)
    A[r:,:,:]=1
    return A
def feature(A,Rad,Mode='contact',unique=0,Limit=30000): 
    S=A.shape;
    h=Rad
    SP=sph(Rad)
    if Mode=='multi':
        A1=shiftmatrix(A,1);
        A2=shiftmatrix(A,2);
        A3=shiftmatrix(A,3);
        T1=(A!=A1)*(A!=A2)*(A!=A3)*(A1!=A2)*(A1!=A3)*(A2!=A3)

        A1=shiftmatrix(A,-1);
        A2=shiftmatrix(A,-2);
        A3=shiftmatrix(A,-3);
        T2=(A!=A1)*(A!=A2)*(A!=A3)*(A1!=A2)*(A1!=A3)*(A2!=A3)

        T=T1+T2
        P=np.argwhere(T>0);
    if Mode=='contact':
        A1=shiftmatrix(A,1);
        A2=shiftmatrix(A,2);
        A3=shiftmatrix(A,3);
        T1=(A!=A1)*(A!=A2)*(A1!=A2)
        T2=(A!=A3)*(A!=A2)*(A3!=A2)
        T3=(A!=A1)*(A!=A3)*(A1!=A3)
        T=T1+T2+T3
        A1=shiftmatrix(A,-1);
        A2=shiftmatrix(A,-2);
        A3=shiftmatrix(A,-3);
        T1=(A!=A1)*(A!=A2)*(A1!=A2)
        T2=(A!=A3)*(A!=A2)*(A3!=A2)
        T3=(A!=A1)*(A!=A3)*(A1!=A3)
        T=T+T1+T2+T3
        P=np.argwhere(T>0);
    if Mode=='corner':
        P=cornerpoints(np.uint8(A==0),Rad)
    omit=(P[:,0]<=h)+(P[:,0]>=(S[0]-h))+(P[:,1]<=h)+(P[:,1]>=(S[1]-h))+(P[:,2]<=h)+(P[:,2]>=(S[2]-h));
    omit=np.argwhere(omit)
    P=np.delete(P,omit,0)

    a=0;
    X=[];
    Locs=[];
    MaxIt=P.shape[0]
    
    if MaxIt>Limit:
        MaxIt=Limit
    if MaxIt<30:
        MaxIt=P.shape[0]
    X0=np.zeros((MaxIt,(h*2+1)**3))
    Locs0=np.zeros((MaxIt,3))
    b=0;
    while (a<MaxIt):
        p=np.random.randint(0,P.shape[0])
        # p=a
        t=A[P[p,0]-h:P[p,0]+h+1,P[p,1]-h:P[p,1]+h+1,P[p,2]-h:P[p,2]+h+1]
        t=t*SP
        if np.isin(-1,t):
            a=a+1;
            continue
        X0[b,:]=np.ndarray.flatten(t)
        Locs0[b,:]=P[p,:]
        b=b+1
        a=a+1;
    X0=np.int8(X0==0) 
    if b==0:
        # print('Tight Angle')
        return [],[]
    X=X0[:b-1,:]
    Locs=Locs0[:b-1,:]
    if unique==1:
        X,p=np.unique(X, axis=0,return_index=True)
        Locs=Locs[p,:]
        if len(p)>20:
            Locs=Locs[:10,...]
            X=X[:10,...]
    INPUT_SHAPE=[-1,Rad*2+1,Rad*2+1,Rad*2+1,1]
    X=np.reshape(X,INPUT_SHAPE)
    return X,Locs

def gen(N,Rad):
    s=41
    X=[]
    Y=[]
    for I in range(N):
        A=np.zeros((s,s,s))
        A[int(s/2+1):,:,:]=1
        Tet=np.random.rand()*170+5
        A=ndi.rotate(A,Tet ,axes=(1,0),order=0,reshape = False)
        A[int(s/2+1):,:,:]=2
        axes2=shuf(np.asarray([0,1,2])); A=np.moveaxis(A, axes2[0], axes2[1])
        A=np.flip(A,axis=np.random.randint(3))
        Tet2=np.random.rand()*180
        A=ndi.rotate(A,Tet2 ,axes=(1,0),order=0,reshape = False)
        A=trim(A,10)
        A[A==0]=3
        A[A==1]=0
        A[A==3]=1
        x,Locs=feature(A,Rad,unique=1)
        if len(x)==0:
            # print('Tight Angle')
            continue
        y=np.ones((x.shape[0],1))*Tet
        X=np.append(X,x)
        Y=np.append(Y,y)
        
        if len(Y)>=N:
            break
        # print(str(int(len(Y)/N*100)) +' % Completed')
    X=np.reshape(X,(-1,(Rad*2+1)**3))
    X=X[:N,:]
    Y=Y[:N]
    INPUT_SHAPE=[-1,Rad*2+1,Rad*2+1,Rad*2+1,1]
    X=np.reshape(X,INPUT_SHAPE)
    return X,Y
def generate(N,Rad,Para=1,regen=1,save=1):
    if regen==0:
        X=np.load('Results/X_Rad_'+str(Rad)+'.npy')
        Y=np.load('Results/Y_Rad_'+str(Rad)+'.npy')
        return X,Y
    if Para:
        X=[]
        Y=[]
        N2=int(np.ceil(N/50))
        Out = Parallel(n_jobs=-1)(delayed(gen)(50,Rad) for k in tqdm(range(1,N2+1)))
        for I in range(N2):
            X=np.append(X,Out[I][0])
            Y=np.append(Y,Out[I][1])
        X=np.reshape(X,(-1,(Rad*2+1)**3))
        X=X[:N,:]
        Y=Y[:N]
        INPUT_SHAPE=[-1,Rad*2+1,Rad*2+1,Rad*2+1,1]
        X=np.reshape(X,INPUT_SHAPE)

    if Para==0:
        X,Y=gen(N,Rad)
        
    if regen:
        if save:
            np.save('Results/X_Rad_'+str(Rad)+'.npy',X)
            np.save('Results/Y_Rad_'+str(Rad)+'.npy',Y)
    return X,Y


def gendrop(r,T): # Generates a droplet on a surface with T contact angle in degree and r radius of the droplet
    A=sph(r)
    A=margin(A,int(r))
    M=(math.cos(T*math.pi/180)+1)/2
    N=int(r+M*2*(r+1)) 
    # N=int(r+M*2*(r)) 
    # if T<90:
    #     N=N-1
    if T>90:
        N=N+1 
    B=A*0; B[N:,:,:]=1;
    A=A*2*(1-B)+B
    return A

def modelmake(INPUT_SHAPE,ModelType=1,Num=10000):
    if ModelType==1: # with conv
        s = Input(INPUT_SHAPE[1:])
        p = Conv3D(16, (3, 3,3), kernel_initializer='he_normal', padding='same') (s)
        p = MaxPooling3D((2, 2,2)) (p)
        p = Dropout(0.5) (p)
        p = Conv3D(32, (3, 3,3), kernel_initializer='he_normal', padding='same') (p)
        p = MaxPooling3D((2, 2,2)) (p)
        p = Dropout(0.4) (p)
        p=BatchNormalization()(p)
        p= Dense(64)(p)
        p= Dense(16)(p)
        p= Dense(4,activation='relu')(p)
    if ModelType==2: #fully connected no conv
        s = Input(INPUT_SHAPE[1:])
        p=Flatten()(s)
        p= Dense(128)(p)
        p = Dropout(0.5) (p)
        p=BatchNormalization()(p)
        p= Dense(64)(p)
        p = Dropout(0.4) (p)
        p=BatchNormalization()(p)
        p= Dense(32)(p)
        p = Dropout(0.3) (p)
        p=BatchNormalization()(p)
        p= Dense(16)(p)
        p= Dense(4,activation='relu')(p)
    if ModelType==3: # shallower no conv
        s = Input(INPUT_SHAPE[1:])
        p=Flatten()(s)
        p= Dense(64)(p)
        p = Dropout(0.5) (p)
        p=BatchNormalization()(p)
        p= Dense(16)(p)
        p= Dense(4,activation='relu')(p)
    p=Flatten()(p)
    p= Dense(1,activation='sigmoid')(p)
    p=Reshape((1,1,1,1))(p)
    model = Model(inputs=[s], outputs=[p])
    epochs=100; num_train_examples=Num; batch_size=50;
    decay_steps = epochs * num_train_examples / batch_size
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=2e-4,
    decay_steps=decay_steps,
    decay_rate=0.9)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mse']) 
        
    return model
def readsec(FileName,n,N):
    f = h5py.File(FileName, 'r')
    length=f['Map'].shape[2]
    p1=np.int32(np.round(np.linspace(0,length,N+1)))[n-1]
    p2=np.int32(np.round(np.linspace(0,length,N+1)))[n]
    if n==N:
        p2=p2+1
    A=f['Map'][:,:,p1:p2]
    f.close()
    print('Section ' +str(n)+ ' out of ' +str(N)+ ' is read.')
    return A
def now():
    import datetime
    d1 = datetime.datetime(1, 1, 1)
    d2 = datetime.datetime.now()
    d=d2-d1
    dd=d.days+d.seconds/(24*60*60)+d.microseconds/(24*60*60*1e6)+367
    return dd 
def nowstr():
    from datetime import datetime
    now = datetime.now()
    return now.strftime("%d-%b-%Y %H.%M.%S") 
def makecallback(ModelName):
    timestr=nowstr()
    LogName='log_'+timestr+'_'+'Model'+ModelName

    with open("Logs/"+LogName+".txt", "wt") as f:
        f.write('# Path to train file: \n')
        f.write('DataName' +'\n')
        f.write('# Start time: \n')
        f.write(timestr +'\n')
        nowstr()
        st='# Training loss'
        spa=' ' * (40-len(st))
        st=st+spa+'Validation loss'
        f.write(st+'\n')
        

        
    class MyCallback(tf.keras.callbacks.Callback):
        def __init__(self):
            self.val_loss_=None
            self.start_time=now()
        def on_batch_end(self, batch, logs=None):
            if self.val_loss_==None:
                self.val_loss_=logs['mse']
            with open("Logs/"+LogName+".txt", "a") as f:
                st=str(logs['mse'])
                spa=' ' * (40-len(st))
                st=st+spa+str(self.val_loss_)
                f.write(st+'\n')
        def on_test_batch_end(self,batch, logs=None):
            self.val_loss_=logs['mse']
            
    callbacks_list = [MyCallback()]
    return callbacks_list
    
def trainmodel(model,X_train,Y_train,X_val,Y_val,batch_size=50,epochs=100,retrain=1,ModelName='Model'):
    Y_val=np.reshape(Y_val,(-1,1,1,1,1))/180
    Y_train=np.reshape(Y_train,(-1,1,1,1,1))/180
    SaveName='Results/'+ModelName+'.h5'
    if retrain:                  
        model.fit(X_train,Y_train,epochs=epochs,batch_size=batch_size,validation_data=(X_val, Y_val),callbacks=makecallback(ModelName))
        model.save_weights(SaveName);
    else:
        model.load_weights(SaveName)
    return model
def correl3(Coordinates,Angles,Rad):
    Angles2=Angles
    C=Coordinates
    for I in range(len(Angles)):
        
        C2=np.tile(C[I,:],(C.shape[0],1))
        D=np.sqrt(np.sum((C-C2)**2,axis=1))
        ID=np.argwhere(D<Rad*2)
        if len(ID)<=1:
            continue
        Angles2[I,0]=np.mean(Angles[ID,0])
        # print(ID)
    return Angles2
def correls(Coordinates,Angles,Rad):
    if len(Angles)>500:
        Coordinates2=Coordinates*0
        Angles2=Angles*0
        batches=np.linspace(0,len(Angles),int(len(Angles)/500+1))
        ids=np.argsort(Coordinates[:,2])
        for I in range(1,len(batches)):
            t1,t2=correl(Coordinates[ids[int(batches[I-1]):int(batches[I])],:],Angles[ids[int(batches[I-1]):int(batches[I])]],Rad)
            Coordinates2[int(batches[I-1]):int(batches[I]),:]=t1
            Angles2[int(batches[I-1]):int(batches[I])]=t2
            
        # Angles2=Angles2[ids]
        # Coordinates2=Coordinates2[ids,:]
    else:
        Coordinates2,Angles2=correl(Coordinates,Angles,Rad)
    return Coordinates2,Angles2
            
        
        
    
    
def correl(Coordinates,Angles,Rad):
    from scipy.spatial import distance
    Angles2=Angles
    C=Coordinates
    for I in range(len(Angles)):
        D=distance.cdist(C, np.reshape(C[I,:],[1,3]), 'cityblock')
        ii=D<(Rad*2)
        if np.sum(ii)<=1:
            continue
        Angles2[I]=np.mean(Angles[ii])
    return Coordinates,Angles2
def shuf(L):

    random.shuffle(L)
    return L
def splitdata(X,Y,Rat=[.8,.1,.1]):
    length=X.shape[0]
    List=np.arange(0,length)
    List=shuf(List)
    N=np.int32([0,length*Rat[0],length*(Rat[0]+Rat[1]),length])
    TrainList=List[N[0]:N[1]]
    ValidList=List[N[1]:N[2]]
    TestList=List[N[2]:N[3]]
    X_train=X[TrainList,...]
    Y_train=Y[TrainList,...]
    X_val=X[ValidList,...]
    Y_val=Y[ValidList,...]
    X_test=X[TestList,...]
    Y_test=Y[TestList,...]
    return X_train,Y_train,X_val,Y_val,X_test,Y_test 
def testmodel(model,X,Y):
    Y2=model.predict(X)*180
    Y,p=np.unique(Y,return_index=True)
    Y2=Y2[p]
    plt.scatter(Y,Y2)
    R2=r2_score(np.ndarray.flatten(Y),np.ndarray.flatten(Y2))
    R2=np.mean(abs(Y-np.ndarray.flatten(Y2)))
    print('R-squared is '+str(np.round(R2,3)))
    return R2
def h5size(Name,Field):
    # Fields is list of hdf file fields
    with h5py.File(Name,'r') as f:
        Shape=f[Field].shape  
    return Shape 


def remout(y_final,P_final,per=.02):
    MIN=np.quantile(y_final,per); MAX=np.quantile(y_final,1-per); 
    if abs(MIN-MAX)<2:
        return y_final, P_final
    p=np.argwhere((y_final>MIN)*(y_final<MAX)); 
    y_final=y_final[p[:,0],...]
    P_final=P_final[p[:,0],...]
    return y_final, P_final
def combines(y1,p1,y2,p2):
    if len(y1)>500:
        y11=y1*1
        p11=p1
        
        batches=np.linspace(0,len(y1),int(len(y1)/500+1))
        batches2=np.linspace(0,len(y2),int(len(y1)/500+1))
        ids=np.argsort(p1[:,2])
        ids2=np.argsort(p2[:,2])
        for I in range(1,len(batches)):
            t1,t2=combine(y1[ids[int(batches[I-1]):int(batches[I])]],p1[ids[int(batches[I-1]):int(batches[I])],:],
                          y2[ids2[int(batches2[I-1]):int(batches2[I])]],p2[ids2[int(batches2[I-1]):int(batches2[I])],:])
            p11[int(batches[I-1]):int(batches[I]),:]=t2
            y11[int(batches[I-1]):int(batches[I])]=t1
            
        # Angles2=Angles2[ids]
        # Coordinates2=Coordinates2[ids,:]
    else:
        y11,p11=combine(y1,p1,y2,p2)
    y11, p11=remout(y11, p11,per=.0001)
    return y11,p11
def combine(y1,p1,y2,p2):
    y11=y1*1
    for I in range(len(y1)):         
        C=np.tile(p1[I,:],(p2.shape[0],1))
        D=np.sqrt(np.sum((C-p2)**2,axis=1))
        ID=np.argwhere(D<32)
        C2=np.tile(p1[I,:],(p1.shape[0],1))
        D2=np.sqrt(np.sum((C2-p1)**2,axis=1))
        ID2=np.argwhere(D2<32)
        # print(ID)
        if len(ID)<=1 or len(ID2)<=1:
            continue
        y11[I,0]=y1[I,0]-(np.mean(y1[ID2,0])-np.mean(y2[ID,0]))
        # p11=p1
        # print(ID)
    # y11, p11=remout(y11, p1,per=.01)
    return y11,p1
def combineorig(y1,p1,y2,p2):
    y11=y1*1
    for I in range(len(y1)):         
        C=np.tile(p1[I,:],(p2.shape[0],1))
        D=np.sqrt(np.sum((C-p2)**2,axis=1))
        ID=np.argwhere(D<32)
        C2=np.tile(p1[I,:],(p1.shape[0],1))
        D2=np.sqrt(np.sum((C2-p1)**2,axis=1))
        ID2=np.argwhere(D2<32)
        # print(ID)
        if len(ID)<=1 or len(ID2)<=1:
            continue
        y11[I,0]=y1[I,0]-(np.mean(y1[ID2,0])-np.mean(y2[ID,0]))
        # print(ID)
    y11, p11=remout(y11, p1,per=.0001)
    return y11,p11
def predict(model,Rad,Array,Para=1,export=None,Mode='contact'):
    if isinstance(Array, str):
        S=h5size(Array,'Map')
    else:
        S=Array.shape
    
    vol=S[0]*S[1]*S[2]
    pieces=int(vol/800**3);
    if pieces==0:
        pieces=1
    print('Sample is divided into '+str(pieces) + ' Pieces')
    def calc(I):
        print('Calculated pieces:'+str(I))
        if isinstance(Array, str):
            A=readsec(Array,I,pieces)
        else:
            A=Array
        x_final,P=feature(A,Rad,Mode=Mode) 
        return x_final,P,A.shape[2]
    x_final=[]
    P_final=[]
    Lag=0
    if Para==1:
        Out = Parallel(n_jobs=-1)(delayed(calc)(k) for k in tqdm(range(1,pieces+1)))
        for I in range(pieces):
            if I>0:
                Lag=Out[I-1][2]+Lag
               
            P=Out[I][1]
            P[:,2]=P[:,2]+Lag
            x_final=np.append(x_final,np.ndarray.flatten(Out[I][0]))
            P_final=np.append(P_final,np.ndarray.flatten(P)) 
    if Para==0:        
        for I in range(1,pieces+1):
            x_final2,P,dum=calc(I)
            x_final=np.append(x_final,x_final2) 
            P_final=np.append(P_final,np.ndarray.flatten(P)) 
    
    INPUT_SHAPE=[-1,Rad*2+1,Rad*2+1,Rad*2+1,1]
    x_final=np.reshape(x_final,INPUT_SHAPE)

    P_final=np.reshape(P_final,(-1,3))  

    y_final=model.predict(x_final)*180
    y_final=np.reshape(y_final,(-1,1))
    y_final, P_final=remout(y_final, P_final,per=.0001)

    t,y_final_cor=correls(P_final,y_final,Rad)
    t,y_final_cor=correls(P_final,y_final_cor,Rad)
    t,y_final_cor=correls(P_final,y_final_cor,Rad)

    return y_final_cor,P_final

def saveresults(Angles,Coordinates,SaveName):
    Results=np.concatenate((Angles,Coordinates),axis=1)
    np.savetxt(SaveName+'.txt', Results, fmt='%8.3f', delimiter=' ', header=' Angle     X        Y        Z')

def testspheres(model,Rad):
    Ang2=np.zeros((10,2))
    for I in range(10):
        print(I)
        FileName='Data/Images/A1_' +str(I+1)+'.h5'
        A=readsec(FileName,1,1)
        X,p=feature(A,Rad)
        Y2=model.predict(X)
        Ang2[I,0]=np.mean(Y2)*180
        # plt.figure()
        # plt.hist(np.squeeze(Dist))
        # plt.show()
        FileName='Data/Images/A2_' +str(I+1)+'.h5'
        A=readsec(FileName,1,1)
        X,p=feature(A,Rad)
        Y2=model.predict(X)
        # Dist=(Y2)*180
        # plt.figure()
        # plt.hist(np.squeeze(Dist))
        # plt.show()
        Ang2[I,1]=np.mean(Y2)*180
    import scipy.io as sio    
    
    B=sio.loadmat('Data/GT.mat')  
    GT=B['GT'] 
    plt.figure()
    plt.scatter(Ang2[:,0],GT[:,0]) 
    plt.scatter(Ang2[:,1],GT[:,1]) 
    plt.plot([1,180],[1,180])
def hybridpredict(FileName,N,Rad,Para=1,regen=0,retrain=0,ModelType=2,Mode='contact'):
    if len(N)==2:
        Rad1=Rad[0] # larger size
        INPUT_SHAPE=[-1,Rad1*2+1,Rad1*2+1,Rad1*2+1,1]
        X,Y=generate(N[0],Rad1,Para=Para,regen=regen)  
        X_train,Y_train,X_val,Y_val,X_test,Y_test=splitdata(X,Y,[.8,.1,.1])
        model1=modelmake(INPUT_SHAPE,ModelType=ModelType,Num=N[0])
        model1=trainmodel(model1,X_train,Y_train,X_val,Y_val,epochs=30,retrain=retrain,ModelName='Model_Type'+str(ModelType)+'_Rad'+str(Rad1))
    
        Rad2=Rad[1] # smaller size
        INPUT_SHAPE=[-1,Rad2*2+1,Rad2*2+1,Rad2*2+1,1]
        X,Y=generate(N[1],Rad2,Para=Para,regen=regen)  
        X_train,Y_train,X_val,Y_val,X_test,Y_test=splitdata(X,Y,[.8,.1,.1])
        model2=modelmake(INPUT_SHAPE,ModelType=ModelType,Num=N[1])
        model2=trainmodel(model2,X_train,Y_train,X_val,Y_val,epochs=30,retrain=retrain,ModelName='Model_Type'+str(ModelType)+'_Rad'+str(Rad2))
    
        Angles1,Coordinates1=predict(model1,Rad1,FileName,Mode=Mode)
        Angles2,Coordinates2=predict(model2,Rad2,FileName,Mode=Mode)
        Angles3,Coordinates3=combines(Angles1,Coordinates1,Angles2,Coordinates2)
        model1.save('Model/M8.h5')
        model2.save('Model/M4.h5')
    if len(N)==1:
        Rad1=Rad[0] # larger size
        INPUT_SHAPE=[-1,Rad1*2+1,Rad1*2+1,Rad1*2+1,1]
        X,Y=generate(N[0],Rad1,Para=Para,regen=regen)  
        X_train,Y_train,X_val,Y_val,X_test,Y_test=splitdata(X,Y,[.8,.1,.1])
        model1=modelmake(INPUT_SHAPE,ModelType=ModelType,Num=N[0])
        model1=trainmodel(model1,X_train,Y_train,X_val,Y_val,epochs=30,retrain=retrain,ModelName='Model_Type'+str(ModelType)+'_Rad'+str(Rad1))
        
        Angles3,Coordinates3=predict(model1,Rad1,FileName)
        
    return Angles3,Coordinates3
def getangle(FileName,Para=1,regen=0,retrain=0,ModelType=2,Mode='contact',Fast=0):
    import keras
    Rad=[8,4];
    N=[10000,10000]
    if Fast==1:
        Rad=[8];
        N=[1000]        
    if len(N)==2:
        Rad1=Rad[0] # larger size
        # INPUT_SHAPE=[-1,Rad1*2+1,Rad1*2+1,Rad1*2+1,1]
        # X,Y=generate(N[0],Rad1,Para=Para,regen=regen)  
        # X_train,Y_train,X_val,Y_val,X_test,Y_test=splitdata(X,Y,[.8,.1,.1])
        # model1=modelmake(INPUT_SHAPE,ModelType=ModelType,Num=N[0])
        # model1=trainmodel(model1,X_train,Y_train,X_val,Y_val,epochs=30,retrain=retrain,ModelName='Model_Type'+str(ModelType)+'_Rad'+str(Rad1))
        model1=keras.models.load_model('Model/M8.h5')
        model2=keras.models.load_model('Model/M4.h5')
        Rad2=Rad[1] # smaller size
        # INPUT_SHAPE=[-1,Rad2*2+1,Rad2*2+1,Rad2*2+1,1]
        # X,Y=generate(N[1],Rad2,Para=Para,regen=regen)  
        # X_train,Y_train,X_val,Y_val,X_test,Y_test=splitdata(X,Y,[.8,.1,.1])
        # model2=modelmake(INPUT_SHAPE,ModelType=ModelType,Num=N[1])
        # model2=trainmodel(model2,X_train,Y_train,X_val,Y_val,epochs=30,retrain=retrain,ModelName='Model_Type'+str(ModelType)+'_Rad'+str(Rad2))
    
        Angles1,Coordinates1=predict(model1,Rad1,FileName,Mode=Mode)
        Angles2,Coordinates2=predict(model2,Rad2,FileName,Mode=Mode)
        Angles3,Coordinates3=combines(Angles1,Coordinates1,Angles2,Coordinates2)
        model1.save('Model/M8.h5')
        model2.save('Model/M4.h5')
    if len(N)==1:
        # Rad1=Rad[0] # larger size
        # INPUT_SHAPE=[-1,Rad1*2+1,Rad1*2+1,Rad1*2+1,1]
        # X,Y=generate(N[0],Rad1,Para=Para,regen=regen)  
        # X_train,Y_train,X_val,Y_val,X_test,Y_test=splitdata(X,Y,[.8,.1,.1])
        # model1=modelmake(INPUT_SHAPE,ModelType=ModelType,Num=N[0])
        # model1=trainmodel(model1,X_train,Y_train,X_val,Y_val,epochs=30,retrain=retrain,ModelName='Model_Type'+str(ModelType)+'_Rad'+str(Rad1))
        Rad1=Rad[0] # larger size
        # INPUT_SHAPE=[-1,Rad1*2+1,Rad1*2+1,Rad1*2+1,1]
        # X,Y=generate(N[0],Rad1,Para=Para,regen=regen)  
        # X_train,Y_train,X_val,Y_val,X_test,Y_test=splitdata(X,Y,[.8,.1,.1])
        # model1=modelmake(INPUT_SHAPE,ModelType=ModelType,Num=N[0])
        # model1=trainmodel(model1,X_train,Y_train,X_val,Y_val,epochs=30,retrain=retrain,ModelName='Model_Type'+str(ModelType)+'_Rad'+str(Rad1))
        model1=keras.models.load_model('Model/M8.h5')
        # model2=keras.models.load_model('Model/M4.h5')
        # Rad2=Rad[1] # smaller size
        # INPUT_SHAPE=[-1,Rad2*2+1,Rad2*2+1,Rad2*2+1,1]
        # X,Y=generate(N[1],Rad2,Para=Para,regen=regen)  
        # X_train,Y_train,X_val,Y_val,X_test,Y_test=splitdata(X,Y,[.8,.1,.1])
        # model2=modelmake(INPUT_SHAPE,ModelType=ModelType,Num=N[1])
        # model2=trainmodel(model2,X_train,Y_train,X_val,Y_val,epochs=30,retrain=retrain,ModelName='Model_Type'+str(ModelType)+'_Rad'+str(Rad2))
    
        Angles3,Coordinates3=predict(model1,Rad1,FileName,Mode=Mode)        
        # Angles3,Coordinates3=predict(model1,Rad1,FileName)
        
    return Angles3,Coordinates3