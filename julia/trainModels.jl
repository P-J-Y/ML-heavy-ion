using Flux
using Flux.Losses: mse
using CUDA
using Plots
using Random
using MAT
using Statistics
#using Dates
using JLD2
#using BenchmarkTools

DataName = "tot"
@load "data\\modelData.jld2" X Xcv Xtest Y Ycv Ytest SWflag SWflagcv SWflagtest
#Y: FetoO O7to6 C6to5 C6to4 nHe2 vHe2
#BData npData vpData vthpData dBrms_30min dVrms_30min sigmaC_30min ssnData F107Data
#0:streamer 1:CH 2:CME 3:unidentified -1:nHe2
#= 现在都是一起训练，不再需要这部分单独读取一种数据
Y = Array(O7to6)
Ycv = Array(O7to6cv)
Ytest = Array(O7to6test)
=#
CUDA.allowscalar(false)