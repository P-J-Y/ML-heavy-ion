# 如果你要一起训练，Y就应该归一化

using Flux
using Flux: train!
using Flux.Losses: mse
using CUDA
using Plots
using Random
using MAT
using Statistics
#using Dates
using JLD2
#using BenchmarkTools

DataName = "O7to6"
@load "data\\modelData.jld2" X Xcv Xtest Y Ycv Ytest SWflag SWflagcv SWflagtest O7to6 O7to6cv O7to6test
#Y: FetoO O7to6 C6to5 C6to4 nHe2 vHe2
#X: BData npData vpData vthpData dBrms_30min dVrms_30min sigmaC_30min ssnData F107Data
#0:streamer 1:CH 2:CME 3:unidentified -1:nHe2
# Y = Array(O7to6)
# Ycv = Array(O7to6cv)
# Ytest = Array(O7to6test)

CUDA.allowscalar(false)

########## model we use, first model has 2 dense layers with BatchNorm, second model has 3 dense layers with BatchNorm ##########

"""
If model has only 1 output feature, use this function to build the model
"""
function buildModel(nx::Int,λ::Float32,minibatchsize::Int)
    local model = Chain(Dense(nx,3*nx),
    BatchNorm(3*nx,leakyrelu),
    Dense(3*nx,nx),
    BatchNorm(nx,leakyrelu),
    Dense(nx,1,leakyrelu)) |> gpu
    local regularParams = Flux.params(model)
    delete!(regularParams,model[1].bias)
    delete!(regularParams,model[3].bias)
    delete!(regularParams,model[5].bias)
    delete!(regularParams,model[2].γ)
    delete!(regularParams,model[2].β)
    delete!(regularParams,model[4].γ)
    delete!(regularParams,model[4].β)
    function calCost(X,Y)
        sqnorm(x) = sum(abs2,x)
        mse(model(X),Y) + λ*sum(sqnorm,regularParams)/minibatchsize
    end
    ps = Flux.params(model)
    delete!(ps,model[1].bias)
    delete!(ps,model[3].bias)
    model,calCost,ps
end

"""
If model has multiple output features, use this function to build the model
"""
function buildModel2(nx::Int,ny::Int,λ::Float32,minibatchsize::Int)
    local model = Chain(Dense(nx,3*nx),
    BatchNorm(3*nx,leakyrelu),
    Dense(3*nx,2*nx),
    BatchNorm(2*nx,leakyrelu),
    Dense(2*nx,nx),
    BatchNorm(nx,leakyrelu),
    Dense(nx,ny)) |> gpu
    local regularParams = Flux.params(model)
    delete!(regularParams,model[1].bias)
    delete!(regularParams,model[3].bias)
    delete!(regularParams,model[5].bias)
    delete!(regularParams,model[7].bias)
    delete!(regularParams,model[2].γ)
    delete!(regularParams,model[2].β)
    delete!(regularParams,model[4].γ)
    delete!(regularParams,model[4].β)
    delete!(regularParams,model[6].γ)
    delete!(regularParams,model[6].β)
    function calCost(X,Y)
        sqnorm(x) = sum(abs2,x)
        mse(model(X),Y;agg=sum)/minibatchsize + λ*sum(sqnorm,regularParams)/minibatchsize
    end
    ps = Flux.params(model)
    delete!(ps,model[1].bias)
    delete!(ps,model[3].bias)
    delete!(ps,model[5].bias)
    model,calCost,ps
end

####### test the model with different hyperparameters #######

"""
For muultiple output models
Test different hyperparameters: learning rate α, regularization parameter λ, and number of hidden units nh
remenber to set the best default values for α, λ, and nh when testing the other two hyperparameters
"""
function testModels(
    theParameterName::String,
    theParameters::Vector{},
    epochNum::Int32,
    X::Array{Float32,2},
    Y::Array{Float32,2},
    Xcv::Array{Float32,2},
    Ycv::Array{Float32,2};
    α=Float32(0.005),
    minibatchsize=1024,
    λ=Float32(0.005),
    buildmymodel = buildModel)

    local Costs = []
    local trainCCs = []
    local cvCosts = []
    local cvCCs = []
    local nx,m = size(X)
    local ny = size(Y,1)
    trainData = Flux.Data.DataLoader(
    (gpu(X), gpu(Y)),
    batchsize=minibatchsize,
    shuffle=true)
    for theParameterIndex in eachindex(theParameters)
        theParameter = theParameters[theParameterIndex]
        println()
        @show theParameterIndex
        if theParameterName=="α"
            α = theParameter
            @show α
            opt = ADAM(α)
            if ny==1
                model,calCost,ps = buildmymodel(nx,λ,minibatchsize)
            else
                model,calCost,ps = buildmymodel(nx,ny,λ,minibatchsize)
            end
        elseif theParameterName=="λ"
            λ = theParameter
            @show λ
            opt = ADAM(α)
            if ny==1
                model,calCost,ps = buildmymodel(nx,λ,minibatchsize)
            else
                model,calCost,ps = buildmymodel(nx,ny,λ,minibatchsize)
            end
        elseif theParameterName=="nh"
            nh = Int32(theParameter)
            @show nh
            opt = ADAM(α)
            model,calCost,ps = buildmymodel(nx,nh,λ,minibatchsize)
        else
            @error "theParameterName must be α,λ,nh"
        end

        local trainCost = []
        local trainCC = []

        for epoch = 1:epochNum
            Flux.train!(calCost, ps, trainData, opt)
            # if epoch % (epochNum ÷ 15) == 1
            if epoch % 5 == 1
                Flux.testmode!(model)
                model_cpu = cpu(model)
                lossTrain = mse(model_cpu(X), Y)
                ccTrain = cor(model_cpu(X)[:], Y[:])
                push!(trainCost,lossTrain)
                push!(trainCC,ccTrain)
                Flux.trainmode!(model)
                @show epoch
                @show lossTrain
                @show ccTrain
            end
        end

        Flux.testmode!(model)
        model_cpu = cpu(model)

        local losscv = mse(model_cpu(Xcv),Ycv)
        local cccv = cor(model_cpu(Xcv)[:],Ycv[:])
        push!(cvCosts,losscv)
        push!(cvCCs,cccv)
        push!(trainCCs,trainCC)
        push!(Costs,trainCost)
        Flux.trainmode!(model)
    end

    Costs,cvCosts,cvCCs,trainCCs
    # Costs,cvCosts
end
#=
theParameterName = "α"
#theParameters = [Float32(10^-1.),]
theParameters = 10 .^(Float32(1)*rand(Float32,2).-Float32(3))
#theParameterName = "λ"

epochNum = Int32(1001)
testCosts,cvCosts,cvCCs,trainCCs = testModels(
# testCosts,cvCosts = testModels(
theParameterName,
theParameters,
epochNum,
X,
Y,
Xcv,
Ycv;
buildmymodel = buildModel,
α=Float32(0.006),
minibatchsize=2048,
λ=Float32(1),
)

Plots.plot(testCosts,
label=round.(theParameters,digits=5)',
xlabel="epoch",
ylabel="mse of train set",
# ylims=(0,100),
ls=:auto)
savefig("figure\\"*
DataName*
"\\"*
"J_"*
theParameterName*
".png")
=#

########## train the model with the best hyperparameters ##########

"""
For single output models
"""
function trainOneModel(
    epochNum::Int32,
    X::Array{Float32,2},
    Y::Array{Float32,2},
    Xcv::Array{Float32,2},
    Ycv::Array{Float32,2};
    α = Float32(0.005),
    minibatchsize = 1024,
    λ = Float32(0.005),
    buildmymodel = buildModel,
)


    local nx,m = size(X)
    local ny = size(Y,1)

    opt = ADAM(α)
    if ny==1
        model, calCost, ps = buildmymodel(nx, λ, minibatchsize)
    else
        model, calCost, ps = buildmymodel(nx,ny,λ,minibatchsize)
    end
    trainData = Flux.Data.DataLoader(
        (gpu(X), gpu(Y)),
        batchsize = minibatchsize,
        shuffle = true,
    )
    for epoch = 1:epochNum
        Flux.train!(calCost, ps, trainData, opt)
        if epoch % (epochNum ÷ 5) == 1
            Flux.testmode!(model)
            model_cpu = cpu(model)
            lossTrain = mse(model_cpu(X), Y)*ny
            ccTrain = cor(model_cpu(X)[:], Y[:])
            Flux.trainmode!(model)
            @show epoch
            @show lossTrain
            @show ccTrain
        end
    end

    Flux.testmode!(model)
    model_cpu = cpu(model)
    local losscv = mse(model_cpu(Xcv), Ycv)*ny
    local cccv = cor(model_cpu(Xcv)[:], Ycv[:])
    # local losstrain = mse(model_cpu(X), Y)*ny
    #local cctrain = cor(model_cpu(X)[:], Y[:])
    model_cpu, cccv, losscv
    # model,losscv
end

#=
epochNum = Int32(1001)

themodel,thelosscv = trainOneModel(epochNum,
X,
Y,
Xcv,
Ycv;
buildmymodel = buildModel,
α=Float32(0.004),
λ=Float32(0.0002),
)

#@load "data\\tot\\models.jld2" themodel1 themodel2 themodel3
#device =
Ŷ = cpu(themodel)(X)
Ŷcv = cpu(themodel)(Xcv)
Ŷtest = cpu(themodel)(Xtest)

CCtrain1 = cor(Y,Ŷ,dims=2)
CCdev1 = cor(Ycv,Ŷcv,dims=2)
CCtest1 = cor(Ytest,Ŷtest,dims=2)
=#

function chooseBestModel(
    epochNum::Int32,
    XCls::Array{Float32,2},
    YCls::Array{Float32,2},
    XcvCls::Array{Float32,2},
    YcvCls::Array{Float32,2};
    α = Float32(0.005),
    minibatchsize = 1024,
    λ = Float32(0.005),
    buildmymodel = buildModel,
)
    bestmodel,bestcccv,bestlosscv = trainOneModel(epochNum,
    XCls,
    YCls,
    XcvCls,
    YcvCls;
    buildmymodel = buildModel,
    α=α,
    λ=λ,
    )
#=
    for i in 1:1
        println()
        @show i
        themodel,thecccv,thelosscv = trainOneModel(epochNum,
        XCls,
        YCls,
        XcvCls,
        YcvCls;
        buildmymodel = buildModel,
        α=α,
        λ=λ,
        )
        if thecccv > bestcccv
            bestlosscv = thelosscv
            bestmodel = themodel
            bestcccv = thecccv
            @show bestlosscv
            @show bestcccv
        end
    end
    =#
    bestmodel,bestcccv
end

# epochNum = Int32(801)
# themodel,thecccv = chooseBestModel(epochNum,
# X,
# Y,
# Xcv,
# Ycv;
# buildmymodel = buildModel,
# α=Float32(0.004),
# λ=Float32(0.0002),
# )

"""
Set best hyperparameters for each output feature models
"""
function getTotModel(X,Y,Xcv,Ycv,αs,λs;epochNum=Int32(1001))
    models = []
    devcvs = []
    k = size(Y,1)
    for dataIndex in 1:k
        println()
        local α=αs[dataIndex]
        local λ=λs[dataIndex]
        @show dataIndex

        themodel,thecccv = chooseBestModel(epochNum,
        X,
        reshape(Y[dataIndex,:],(1,:)),
        Xcv,
        reshape(Ycv[dataIndex,:],(1,:));
        buildmymodel = buildModel,
        α=α,
        λ=λ,
        )
        @show thecccv
        push!(models,themodel)
        push!(devcvs,thecccv)
    end
    models,devcvs
end

################# main ########################
# minibatchsize = 1024
αs=Float32.([0.0004,0.005,0.006,0.005,0.005,0.0025])
λs=Float32.([1.,0.4,0.1,0.05,1e-4,0.9])
models,devcvs = getTotModel(X,Y,Xcv,Ycv,αs,λs;epochNum=Int32(1001))

@save "data\\models.jld2" models devcvs
# @load "data\\models.jld2" models
outputNames = ["FetoO", "O7toO6", "C6toC5", "C6toC4", "nHe2", "vHe2"]
outputTitles = ["Fe/O", "O7/O6", "C6/C5", "C6/C4", "Nα", "Vα"]
panelIdxs = ["a" "b" "c" "d" "e" "f"]

#计算每个模型训练集、开发集和测试集的相关系数，并输入到一个表格中
CCs = Matrix{Float32}(undef,6,3)
for i in eachindex(models)
    model = models[i]
    Ŷ = model(X)
    Ŷcv = model(Xcv)
    Ŷtest = model(Xtest)
    cctrain = round(cor(Ŷ[:],Y[i,:]),digits=3)
    cccv = round(cor(Ŷcv[:],Ycv[i,:]),digits=3)
    cctest = round(cor(Ŷtest[:],Ytest[i,:]),digits=3)
    CCs[i,:] = [cctrain,cccv,cctest]
end
# output the table
using DataFrames
df = DataFrame((CCs=outputNames,train=CCs[:,1],dev=CCs[:,2],test=CCs[:,3]))
using CSV
CSV.write("CCs.csv",df)

# hyperparameters
df = DataFrame(((hyperparameters=outputNames,learningRate=αs,lambda=λs,minibandsize=[1024,1024,1024,1024,1024,1024])))
CSV.write("hyperparameters.csv",df)

############## test model without BatchNorm ##############
function buildModelNoBatchNorm(nx::Int,λ::Float32,minibatchsize::Int)
    local model = Chain(Dense(nx,3*nx,leakyrelu),
    Dense(3*nx,nx,leakyrelu),
    Dense(nx,1,leakyrelu)) |> gpu
    function calCost(X,Y)
        sqnorm(x) = sum(abs2,x)
        local regularParams = Flux.params(model)
        mse(model(X),Y) + λ*sum(sqnorm,regularParams)/minibatchsize
    end
    ps = Flux.params(model)
    delete!(ps,model[1].bias)
    delete!(ps,model[2].bias)
    model,calCost,ps
end

theParameterName = "α"
theParameters = [0.00004, 0.0004, 0.004, 0.02]
epochNum = Int32(1001)

testCosts_noBN,cvCosts_noBN,cvCCs_noBN,trainCCs_noBN = testModels(
theParameterName,
theParameters,
epochNum,
X,
reshape(Y[2,:],(1,:)),
Xcv,
reshape(Ycv[2,:],(1,:));
buildmymodel = buildModelNoBatchNorm,
α=Float32(0.004),
minibatchsize=1024,
λ=Float32(0.4),
)

testCosts_BN,cvCosts_BN,cvCCs_BN,trainCCs_BN = testModels(
theParameterName,
theParameters,
epochNum,
X,
reshape(Y[2,:],(1,:)),
Xcv,
reshape(Ycv[2,:],(1,:));
buildmymodel = buildModel,
α=Float32(0.004),
minibatchsize=1024,
λ=Float32(0.4),
)

epochs = 1:5:epochNum

labels = Matrix{String}(undef,1,4)
for i in 1:4
    labels[i] = "α="*string(round(theParameters[i],digits=5))
end
a = Plots.plot(epochs,testCosts_noBN,
label=labels,
xlabel="epoch",
ylabel="mse of train set",
ylims=(0.035,0.05),
ls=:auto,
title="without BatchNorm")
b = Plots.plot(epochs,testCosts_BN,
label=false,
xlabel="epoch",
ylabel="mse of train set",
ylims=(0.035,0.05),
ls=:auto,
title="with BatchNorm")
c = Plots.plot(epochs,trainCCs_noBN,
label=false,
xlabel="epoch",
ylabel="CC of train set",
ylims=(0.5,0.8),
ls=:auto)
d = Plots.plot(epochs,trainCCs_BN,
label=false,
xlabel="epoch",
ylabel="CC of train set",
ylims=(0.5,0.8),
ls=:auto)
Plots.plot(a,b,c,d,layout=(2,2),size=(1000,800))

savefig("figure\\"*
"withoutBN\\"*"J_"*
theParameterName*
".png")

############### train 10 times to show the reproducibility ##############
αs=Float32.([0.0004,0.005,0.006,0.005,0.005,0.0025])
λs=Float32.([1.,0.4,0.1,0.05,1e-4,0.9])

CCs = Array{Float32}(undef,6,3,10)
for i in 1:10
    println()
    @show i
    models,devcvs = getTotModel(X,Y,Xcv,Ycv,αs,λs;epochNum=Int32(1001))
    for j in 1:6
        model = models[j]
        Ŷ = model(X)
        Ŷcv = model(Xcv)
        Ŷtest = model(Xtest)
        cctrain = round(cor(Ŷ[:],Y[j,:]),digits=3)
        cccv = round(cor(Ŷcv[:],Ycv[j,:]),digits=3)
        cctest = round(cor(Ŷtest[:],Ytest[j,:]),digits=3)
        CCs[j,:,i] = [cctrain,cccv,cctest]
    end
end

CC_avg = mean(CCs,dims=3)
CC_std = std(CCs,dims=3)
CC_avg = reshape(CC_avg,(6,3))
CC_std = reshape(CC_std,(6,3))
# output the table
CC_table = Matrix{String}(undef,6,3)
for i in 1:6
    for j in 1:3
        CC_table[i,j] = string(round(CC_avg[i,j], digits=3))*"±"*string(round(CC_std[i,j], digits=3))
    end
end

using CSV
using DataFrames
df = DataFrame((CCs=outputNames,train=CC_table[:,1],dev=CC_table[:,2],test=CC_table[:,3]))
CSV.write("CCs_10times.csv",df)