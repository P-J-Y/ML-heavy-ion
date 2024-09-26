# load the SolO data and predict the output using the trained models

using Plots
using Random
using MAT
using Statistics
using Dates
using JLD2
using Flux
function featureScalling(X)
    (X.-mean(X,dims=2))./std(X,dims=2)
end



###############load ACE data for feature scalling###################
@load "data\\modelData.jld2"
ACEXData = OriginalXYDATA[1:9,:]#BData npData vpData vthpData dBrms_30min dVrms_30min sigmaC_30min ssnData F107Data
ACEXMean = mean(ACEXData,dims=2)
ACEXStd = std(ACEXData,dims=2)

###############SolO Data###################
Data = matread("data\\evalSolOData_modified.mat")
Data = Data["DATA"]'
XData = Data[[1,7,3,6,2,4,5,8,9],:]    #BData npData vpData vthpData dBrms_30min dVrms_30min sigmaC_30min ssnData F107Data
XDataScaled = (XData.-ACEXMean)./ACEXStd
SimpleXDataScaled = XDataScaled[[3,],:] # B np vp (sigmac)
datetimeNum = reshape(Data[end,:],(1,:))
m = size(XData,2)

###############load trained models###################
@load "data\\models.jld2" # 6 models for 6 outputs (FetoO O7to6 C6to5 C6to4 nHe2 vHe2)

m = size(XData,2)
Ŷ = zeros(length(models),m)
for dataIndex in eachindex(models)
    model = models[dataIndex]
    Ŷ[dataIndex,:] = model(cpu(XDataScaled))
end

YData = [Ŷ;XData;datetimeNum]
file = matopen("SolOY_modified.mat", "w")
write(file, "YData", YData)
close(file)

