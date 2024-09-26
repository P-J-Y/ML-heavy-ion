clear;clc;

miu_0 = 4e-7*pi;
m_p = 1.672621637e-27;  %Kg

MagFilePath = 'Data/original/tot/ac_h0s_mfi_*.cdf';
MagFiles = dir(MagFilePath);
IonFilePath = 'Data/original/tot/ac_h2s_swi_*.cdf';
IonFiles = dir(IonFilePath);
ProtonFilePath = 'Data/original/tot/ac_h6s_swi_*.cdf';
ProtonFiles = dir(ProtonFilePath);
SWFilePath = 'Data/original/tot/ac_h0s_swe_*.cdf';
SWFiles = dir(SWFilePath);
OMNIFilePath = 'Data/original/tot/omni2_h0s_mrg1hr_*.cdf';
OMNIFiles = dir(OMNIFilePath);

FilesNum = length(OMNIFiles);

DATA = [];
EPOCH = [];

for theFileNum = 1:FilesNum
    theFileNum
    theMagFile = [MagFiles(theFileNum).folder,'/',MagFiles(theFileNum).name];
    theMagFileInfo = spdfcdfinfo(theMagFile);
    theMagData = spdfcdfread(theMagFile);
    theIonFile = [IonFiles(theFileNum).folder,'/',IonFiles(theFileNum).name];
    theIonFileInfo = spdfcdfinfo(theIonFile);
    theIonData = spdfcdfread(theIonFile);
    theProtonFile = [ProtonFiles(theFileNum).folder,'/',ProtonFiles(theFileNum).name];
    theProtonFileInfo = spdfcdfinfo(theProtonFile);
    theProtonData = spdfcdfread(theProtonFile);
    theSWFile = [SWFiles(theFileNum).folder,'/',SWFiles(theFileNum).name];
    theSWFileInfo = spdfcdfinfo(theSWFile);
    theSWData = spdfcdfread(theSWFile);
    theOMNIFile = [OMNIFiles(theFileNum).folder,'/',OMNIFiles(theFileNum).name];
    theOMNIFileInfo = spdfcdfinfo(theOMNIFile);
    theOMNIData = spdfcdfread(theOMNIFile);
    
    
    MagDataDateTimeNum = theMagData{1};
    B_Magnetitude = theMagData{2};
    B_MagnetitudeNaNPoints = B_Magnetitude<0;
    B_GSE = theMagData{3};
    B_GSENaNPoints = (sum(B_GSE<-100,2)~=0);
    %dBrms = theMagData{4};%scale
    %dBrms(dBrms<0) = NaN;
    pos_GSE = theMagData{4};
    theMagNaNPoints = B_GSENaNPoints | B_MagnetitudeNaNPoints;
    MagDataDateTimeNum = MagDataDateTimeNum(~theMagNaNPoints);
    B_Magnetitude = B_Magnetitude(~theMagNaNPoints);
    B_GSE = B_GSE(~theMagNaNPoints,:);
    pos_GSE = pos_GSE(~theMagNaNPoints,:);
    MagDataDateTime = datetime(MagDataDateTimeNum,'ConvertFrom','datenum');
    pos = pos_GSE(:,1);
    
    IonDataDateTimeNum = theIonData{1};
    nHe2 = theIonData{2};
    nHe2NaNPoints = nHe2<0;
    vHe2 = theIonData{3};
    vHe2NaNPoints = vHe2<0;
    C6to4 = theIonData{5};
    C6to4NaNPoints = C6to4<0;
    C6to5 = theIonData{6};
    C6to5NaNPoints = C6to5<0;
    O7to6 = theIonData{7};
    O7to6NaNPoints = O7to6<0;
    FetoO = theIonData{8};
    FetoONaNPoints = FetoO<0;
    SWType = theIonData{9};   %0:streamer 1:CH 2:CME 3:unidentified -1:nHe2有问题
    IonNaNPoints = nHe2NaNPoints | vHe2NaNPoints | C6to4NaNPoints | C6to5NaNPoints | O7to6NaNPoints | FetoONaNPoints;
    IonDataDateTimeNum = IonDataDateTimeNum(~IonNaNPoints);
    nHe2 = nHe2(~IonNaNPoints);
    vHe2 = vHe2(~IonNaNPoints);
    C6to4 = C6to4(~IonNaNPoints);
    C6to5 = C6to5(~IonNaNPoints);
    O7to6 = O7to6(~IonNaNPoints);
    FetoO = FetoO(~IonNaNPoints);
    SWType = SWType(~IonNaNPoints);
    IonDataDateTime = datetime(IonDataDateTimeNum,'ConvertFrom','datenum');
    
    
    
    ProtonDataDateTimeNum = theProtonData{1};
    nH = theProtonData{2};
    nHNaNPoints = nH<0;
    vH = theProtonData{3};  %速率
    vHNaNPoints = vH<0;
    vthH = theProtonData{4};
    vthHNaNPoints = vthH<0;
    ProtonNaNPoints = nHNaNPoints | vHNaNPoints | vthHNaNPoints;
    ProtonDataDateTimeNum = ProtonDataDateTimeNum(~ProtonNaNPoints);
    nH = nH(~ProtonNaNPoints);
    vH = vH(~ProtonNaNPoints);
    vthH = vthH(~ProtonNaNPoints);
    ProtonDataDateTime = datetime(ProtonDataDateTimeNum,'ConvertFrom','datenum');
    
    SWDataDateTimeNum = theSWData{1};   %分辨率 64s
    Np_SW = theSWData{2};
    Np_SWNaNPoints = Np_SW<0;
    Vp_SW = theSWData{3};  %速率
    Vp_SWNaNPoints = Vp_SW<0;
    Vp_GSE = theSWData{4};
    Vp_GSENaNPoints = (sum(Vp_GSE<-5000,2)~=0);
    SWNaNPoints = Np_SWNaNPoints | Vp_SWNaNPoints | Vp_GSENaNPoints;
    SWDataDateTimeNum = SWDataDateTimeNum(~SWNaNPoints);
    Np_SW = Np_SW(~SWNaNPoints);
    Vp_SW = Vp_SW(~SWNaNPoints);
    Vp_GSE = Vp_GSE(~SWNaNPoints,:);
    SWDataDateTime = datetime(SWDataDateTimeNum,'ConvertFrom','datenum');
    
    ssn = theOMNIData{4};
    F107 = theOMNIData{5};
    DateTimeSolar = theOMNIData{6};
    F107NANPoints = F107>999;
    ssn = ssn(~F107NANPoints);
    F107 = F107(~F107NANPoints);
    DateTimeSolar = DateTimeSolar(~F107NANPoints);
    
    %计算IonDataDateTime 对应时间点前后半小时（共1小时）内，磁场、速度的相关系数（GSE坐标系）
    
    theDataNumM = length(IonDataDateTimeNum);
    sigmaC_30min = zeros(theDataNumM,1); % m * (x y z)
    dBrms_30min = zeros(theDataNumM,1);
    dVrms_30min = zeros(theDataNumM,1);
   parfor i = 1:theDataNumM    %单位是天
        theTimeNumi = IonDataDateTimeNum(i);
        
        theLogicSWTime_30min = SWDataDateTimeNum>(theTimeNumi-1/48) & ...
            SWDataDateTimeNum<(theTimeNumi+1/48);
        theSWTimeNum_30min = SWDataDateTimeNum(theLogicSWTime_30min);
        theVp_GSE30min = Vp_GSE(theLogicSWTime_30min,:);
        theB_GSE30min = interp1(MagDataDateTimeNum,B_GSE,theSWTimeNum_30min);

        theNp = mean(Np_SW(theLogicSWTime_30min));
        theVA = 1e-15*theB_GSE30min./sqrt(miu_0*m_p*theNp);
        
        sigmaC_30min(i) = 2*abs (mean( sum( (theVp_GSE30min-mean(theVp_GSE30min,1,'omitnan')).*(theVA-mean(theVA,1,'omitnan')) ,2,'omitnan') ,'omitnan') ...
            /(mean( sum( (theVp_GSE30min-mean(theVp_GSE30min,1,'omitnan')).*(theVp_GSE30min-mean(theVp_GSE30min,1,'omitnan')) ,2,'omitnan') ,'omitnan') ...
            + mean( sum( (theVA-mean(theVA,1,'omitnan')).*(theVA-mean(theVA,1,'omitnan')) ,2,'omitnan') ,'omitnan')));

        dVrms_30min(i) = sqrt(sum(var(theVp_GSE30min,1,1,'omitnan')));
        
        theLogicMagTime_30min = MagDataDateTimeNum>(theTimeNumi-1/48) & ...
            MagDataDateTimeNum<(theTimeNumi+1/48);
        theMagGSE_30min = B_GSE(theLogicMagTime_30min,:);
        dBrms_30min(i) = sqrt(sum(var(theMagGSE_30min,1,1,'omitnan')));
        
    end
    
    
    
    
    %插值得到一些时间对应的数据点
    theDateTimeNum = IonDataDateTimeNum;
    BData = interp1(MagDataDateTimeNum,B_Magnetitude,theDateTimeNum);
    posData = interp1(MagDataDateTimeNum,pos,theDateTimeNum);
    npData = interp1(ProtonDataDateTimeNum,nH,theDateTimeNum);
    vpData = interp1(ProtonDataDateTimeNum,vH,theDateTimeNum);
    vthpData = interp1(ProtonDataDateTimeNum,vthH,theDateTimeNum);
    ssnData = interp1(DateTimeSolar,double(ssn),theDateTimeNum);
    F107Data = interp1(DateTimeSolar,F107,theDateTimeNum);
     
    theData = [BData posData npData vpData vthpData dBrms_30min dVrms_30min sigmaC_30min ssnData F107Data FetoO O7to6 C6to5 C6to4 nHe2 vHe2 single(SWType)];
    theDataNaNPoints = (sum(isnan(theData),2)~=0);
    theData = theData(~theDataNaNPoints,:);
    
    theDatetime = theDateTimeNum(~theDataNaNPoints);
    DATA = [DATA;theData]; %single
    EPOCH = [EPOCH;theDatetime]; %double
end

save Data/DataTot.mat DATA EPOCH

