# load ACE data, generate train set, cv set and test set

using MAT
using Random
using Statistics
using JLD2
function featureScalling(X)
    (X.-mean(X,dims=2))./std(X,dims=2)
end

Data = matread("data\\DataTot.mat")
DatetimeNum = Data["EPOCH"]'    #datetime
DatetimeNum = DatetimeNum[1,:]
Data = Data["DATA"]'
XData = Data[[1,3,4,5,6,7,8,9,10],:]    #BData npData vpData vthpData dBrms_30min dVrms_30min sigmaC_30min ssnData F107Data
nx = size(XData,1)
YData = Data[11:16,:]   #FetoO O7to6 C6to5 C6to4 nHe2 vHe2
ny = size(YData,1)
SWflag = Data[17,:]    #0:streamer 1:CH 2:CME 3:unidentified -1:nHe2

OriginalXDATA = XData
OriginalXYDATA = [XData;YData]
OriginalSWflag = SWflag
XData = featureScalling(XData)
Data = [XData;YData;SWflag';OriginalXDATA]
m_total = size(Data,2)

# trainset cvset testset 0.8 0.1 0.1
m = floor(Int,m_total * 0.8)
mCV = floor(Int,m_total * 0.1)
mTest = m_total - m - mCV

########## 1. all data random ##########
# Data = Data[:, randperm(m_total)]
# trainData = Data[:, 1:m]
# cvData = Data[:, m+1:m+mCV]
# testData = Data[:, m+mCV+1:end]

########## 2. test set is the last 10% of the data，not random ##########
testData = Data[:, m+mCV+1:end]
Data = Data[:, 1:m+mCV]
randindex = randperm(m+mCV)
Data = Data[:, randindex]
trainData = Data[:, 1:m]
cvData = Data[:, m+1:m+mCV]
##########################################

DatetimeNumtest = reshape(DatetimeNum[m+mCV+1:end],(1,mTest))
DatetimeNum = DatetimeNum[randindex]
DatetimeNumcv = reshape(DatetimeNum[m+1:m+mCV],(1,mCV))
DatetimeNum = reshape(DatetimeNum[1:m],(1,m))


X = trainData[1:nx, :]
Y = trainData[nx+1:nx+ny, :] #FetoO O7to6 C6to5 C6to4 nHe2 vHe2
SWflag = reshape(trainData[nx+ny+1, :],(1,m))
OriginalX = trainData[nx+ny+2:end, :]

Xcv = cvData[1:nx, :]
Ycv = cvData[nx+1:nx+ny, :]
SWflagcv = reshape(cvData[nx+ny+1, :],(1,mCV))
OriginalXcv = cvData[nx+ny+2:end, :]

Xtest = testData[1:nx, :]
Ytest = testData[nx+1:nx+ny, :]
SWflagtest = reshape(testData[nx+ny+1, :],(1,mTest))
OriginalXtest = testData[nx+ny+2:end, :]

FetoO = Y[1,:]'
O7to6 = Y[2,:]'
C6to5 = Y[3,:]'
C6to4 = Y[4,:]'
nHe2 = Y[5,:]'
vHe2 = Y[6,:]'
FetoOcv = Ycv[1,:]'
O7to6cv = Ycv[2,:]'
C6to5cv = Ycv[3,:]'
C6to4cv = Ycv[4,:]'
nHe2cv = Ycv[5,:]'
vHe2cv = Ycv[6,:]'
vp = OriginalX[3,:]'
ssn = OriginalX[8,:]'
F107 = OriginalX[9,:]'
vpcv = OriginalXcv[3,:]'
ssncv = OriginalXcv[8,:]'
F107cv = OriginalXcv[9,:]'
FetoOtest = Ytest[1,:]'
O7to6test = Ytest[2,:]'
C6to5test = Ytest[3,:]'
C6to4test = Ytest[4,:]'
nHe2test = Ytest[5,:]'
vHe2test = Ytest[6,:]'
@save "data\\modelData.jld2" X Y OriginalXYDATA OriginalSWflag OriginalX Xcv Ycv OriginalXcv Xtest Ytest OriginalXtest FetoO O7to6 C6to5 C6to4 nHe2 vHe2 FetoOcv O7to6cv C6to5cv C6to4cv nHe2cv vHe2cv SWflag SWflagtest SWflagcv DatetimeNum DatetimeNumcv DatetimeNumtest vp vpcv ssn ssncv F107 F107cv FetoOtest C6to5test C6to4test O7to6test nHe2test vHe2test