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

"""
a simple model with one hidden layer
""" 
function buildModel1(nx::Int64,nh::Int64,λ::Float32,minibatchsize::Int64)
    model = Chain(Dense(nx,nh),
    Flux.BatchNorm(nh,leakyrelu),
    Dense(nh,1,leakyrelu)) |> gpu
    local regularParams = Flux.params(model)
    delete!(regularParams,model[1].bias)
    delete!(regularParams,model[3].bias)
    delete!(regularParams,model[2].γ)
    delete!(regularParams,model[2].β)
    function calCost(X,Y)
        sqnorm(x) = sum(abs2,x)
        mse(model(X),Y) + λ*sum(sqnorm,regularParams)/minibatchsize
    end
    ps = Flux.params(model)
    delete!(ps,model[1].bias)
    model,calCost,ps
end

function trainOneModel(
    epochNum::Int32,
    X::Array{Float32,2},
    Y::Array{Float32,2},
    Xcv::Array{Float32,2},
    Ycv::Array{Float32,2},
    n1;
    α = Float32(0.005),
    minibatchsize = 1024,
    λ = Float32(0.005),
    buildmymodel = buildModel1,
)


    local nx,m = size(X)
    local ny = size(Y,1)

    opt = ADAM(α)
    if ny==1
        model, calCost, ps = buildmymodel(nx, n1,λ, minibatchsize)
    else
        model, calCost, ps = buildmymodel(nx, ny,n1,λ, minibatchsize)
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
            lossTrain = mse(model_cpu(X), Y)
            ccTrain = cor(model_cpu(X)[:], Y[:])
            Flux.trainmode!(model)
            @show epoch
            @show lossTrain
            @show ccTrain
        end
    end

    Flux.testmode!(model)
    model_cpu = cpu(model)
    local losscv = mse(model_cpu(Xcv), Ycv)
    local cccv = cor(model_cpu(Xcv)[:], Ycv[:])
    local losstrain = mse(model_cpu(X), Y)
    local cctrain = cor(model_cpu(X)[:], Y[:])
    model, cccv, losscv, cctrain,losstrain
end


"""
train a model with only one input feature to show the contribution of this feature
n1s: the number of nodes in the hidden layer
"""
function contribution(X, Y, Xcv, Ycv, n1s::Vector{Int},α,λ)
    nx = size(X, 1)

    CostTrainData = zeros(nx,length(n1s))
    CCTrainData = copy(CostTrainData)
    CostCVData = copy(CostTrainData)
    CCCVData = copy(CostTrainData)
    # Xindexs = collect(1:nx)
    for indexX = 1:nx
        println()
        println("index X=",indexX)

        theX = reshape(X[indexX, :], (1, :))
        theXcv = reshape(Xcv[indexX, :], (1, :))
        theCostTrain = zeros(length(n1s))
        theCCTrain = zeros(length(n1s))
        theCostCV = zeros(length(n1s))
        theCCCV = zeros(length(n1s))
        for i in eachindex(n1s)

            n1 = n1s[i]
            @show n1
            ~, theCCCV[i], theCostCV[i], theCCTrain[i],theCostTrain[i]=
            trainOneModel(
                Int32(1001),
                theX,
                Y,
                theXcv,
                Ycv,
                n1;
                α = α,
                minibatchsize = 1024,
                λ = λ,
                buildmymodel = buildModel1,
            )
        end
        CostTrainData[indexX,:] = theCostTrain
        CostCVData[indexX,:] = theCostCV
        CCTrainData[indexX,:] = theCCTrain
        CCCVData[indexX,:] = theCCCV
    end

    CostTrainData,CCTrainData,CostCVData,CCCVData
end
#test contribution

function contributionTol(X,Y,Xcv,Ycv,αs,λs,n1s;epochNum=Int32(1001))
    ny = length(αs)
    MSETrain = []
    MSEDev = []
    CCTrain = []
    CCDev = []
    for Yindex in 1:ny
        @show Yindex
        CostTrainData,CCTrainData,CostCVData,CCCVData = contribution(X,
        reshape(Y[Yindex,:],(1,:)),
        Xcv,
        reshape(Ycv[Yindex,:],(1,:)),
        n1s,
        αs[Yindex],
        λs[Yindex],
        )
        push!(MSETrain,CostTrainData)
        push!(MSEDev,CostCVData)
        push!(CCTrain,CCTrainData)
        push!(CCDev,CCCVData)
    end
    MSETrain,MSEDev,CCTrain,CCDev
end

################ 这部分画contribution##############
n1s = [40,70,100]
αs=Float32.([0.0004,0.005,0.006,0.005,0.005,0.0025])
λs=Float32.([1.,0.4,0.1,0.05,1e-4,0.9])
MSETrain,MSEDev,CCTrain,CCDev = contributionTol(X,Y,Xcv,Ycv,αs,λs,n1s;epochNum=Int32(1001))
# @save "data\\contributions.jld2" MSETrain MSEDev CCTrain CCDev

@load "data\\contributions.jld2" MSETrain MSEDev CCTrain CCDev
outputNames = ["FetoO" "O7toO6" "C6toC5" "C6toC4" "nHe2" "vHe2"]
outputTitles = ["Fe/O" "O7/O6" "C6/C5" "C6/C4" "Nα" "Vα"]
inputNames = ["B" "Np" "Vp" "Vthp" "δBrms" "δVrms" "σc" "SSN" "F10.7"]
panelidxs = ["a" "b" "c" "d" "e" "f"]
devs = []
trains = []
devL = @layout grid(3,4)
trainL = @layout grid(3,4)

for i in eachindex(outputNames)

    theMSETrain = MSETrain[i]
    theMSEDev = MSEDev[i]
    theCCTrain = CCTrain[i]
    theCCDev = CCDev[i]

    # l = @layout [a b]

    Pcost = Plots.plot(n1s,theMSETrain';label=inputNames,marker=5,
    xlabel=((i==6 || i==5) ? "Node number" : ""),
    ylabel="MSE",
    xticks=n1s,
    title="("*panelidxs[i]*"1)  "*outputTitles[i]*" (train set)",
    legend= (i==1 ? :bottom : nothing),
    shape=:auto,
    #ylim=(0.002,0.003),
    )
    Pcc = Plots.plot(n1s,theCCTrain';label=inputNames,marker=5,
    xlabel=((i==6 || i==5) ? "Node number" : ""),
    ylabel="CC",
    xticks=n1s,
    title="("*panelidxs[i]*"2)  "*outputTitles[i]*" (train set)",legend=(i==1 ? :bottom : nothing),
    #ylim=(0,0.5),
    shape=:auto,
    )
    # ptrain = Plots.plot(Pcc,Pcost,layout=l)
    push!(trains,Pcost,Pcc)

    # savefig("figure\\paper\\contributions\\"*outputNames[i]*"CostandCCTrain.png")

    # l = @layout [a b]
    Pcost = Plots.plot(n1s,theMSEDev';label=inputNames,marker=5,
    xlabel=((i==6 || i==5) ? "Node number" : ""),
    ylabel="MSE",
    title="("*panelidxs[i]*"1)   "*outputTitles[i],
    xticks=n1s,
    legend=(i==1 ? :bottom : nothing),
    shape=:auto,
    #ylim=(0.002,0.003),
    )
    Pcc = Plots.plot(n1s,theCCDev';label=inputNames,marker=5,
    xlabel=((i==6 || i==5) ? "Node number" : ""),
    ylabel="CC",
    xticks=n1s,
    title="("*panelidxs[i]*"2)   "*outputTitles[i],
    legend=(i==1 ? :bottom : nothing),
    #ylim=(0,0.5),
    shape=:auto,
    )
    # pdev = Plots.plot(Pcc,Pcost,layout=l)
    push!(devs,Pcost,Pcc)
    # savefig("figure\\paper\\contributions\\"*outputNames[i]*"CostandCCDev.png")
end

plot(devs...,layout=devL,size=(1000,850))
savefig("figure\\contributions\\allDev.png")
