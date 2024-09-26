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


#Plots.scatter(year.(DatetimeNum'),F107',label="ssn")
#savefig("figure\\test\\F107DateTime.png")

#=
meanY = mean([Y Ycv Ytest],dims=2)
stdY = std([Y Ycv Ytest],dims=2)
Yscaled = (Y.-meanY)./stdY
Yscaledcv = (Ycv.-meanY)./stdY
Yscaledtest = (Ytest.-meanY)./stdY
#using CH Data
X1 = X[:,findall(SWflag'.==1)]
Y1 = Yscaled[1:4,findall(SWflag'.==1)]
X1cv = Xcv[:,findall(SWflagcv'.==1)]
Y1cv = Yscaledcv[1:4,findall(SWflagcv'.==1)]
X1test = Xtest[:,findall(SWflagtest'.==1)]
Y1test = Yscaledtest[1:4,findall(SWflagtest'.==1)]
#using streamer Data
X2 = X[:,findall(SWflag'.==0)]
Y2 = Yscaled[1:4,findall(SWflag'.==0)]
X2cv = Xcv[:,findall(SWflagcv'.==0)]
Y2cv = Yscaledcv[1:4,findall(SWflagcv'.==0)]
X2test = Xtest[:,findall(SWflagtest'.==0)]
Y2test = Yscaledtest[1:4,findall(SWflagtest'.==0)]
#using CME Data
X3 = X[:,findall(SWflag'.==2)]
Y3 = Yscaled[1:4,findall(SWflag'.==2)]
X3cv = Xcv[:,findall(SWflagcv'.==2)]
Y3cv = Yscaledcv[1:4,findall(SWflagcv'.==2)]
X3test = Xtest[:,findall(SWflagtest'.==2)]
Y3test = Yscaledtest[1:4,findall(SWflagtest'.==2)]
=#

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

function buildTestModel2(nx::Int64,λ::Float32,minibatchsize::Int64)
    local model = Chain(Dense(nx,3*nx),
    BatchNorm(3*nx,leakyrelu),
    Dense(3*nx,nx),
    BatchNorm(nx,leakyrelu),
    Dense(nx,1,leakyrelu)) |> gpu
    local regularParams = Flux.params(model)
    delete!(regularParams,model[1].b)
    delete!(regularParams,model[3].b)
    delete!(regularParams,model[5].b)
    delete!(regularParams,model[2].γ)
    delete!(regularParams,model[2].β)
    delete!(regularParams,model[4].γ)
    delete!(regularParams,model[4].β)
    function calCost(X,Y)
        sqnorm(x) = sum(abs2,x)
        mse(model(X),Y) + λ*sum(sqnorm,regularParams)/minibatchsize
    end
    ps = Flux.params(model)
    delete!(ps,model[1].b)
    delete!(ps,model[3].b)
    model,calCost,ps
end


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
model for train together
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


"""
记得改：
默认的学习率、正则化系数
不同模型参数类型可能要改
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
    #local trainCCs = []
    local cvCosts = []
    #local cvCCs = []
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
        #=
        for epoch in 1:epochNum
            Flux.train!(calCost,ps,trainData,opt)
            if epoch%(epochNum÷20)==1
                Flux.testmode!(model)
                model_cpu = cpu(model)
                lossTrain = mse(model_cpu(X),Y;agg=sum)/m
                push!(trainCost,lossTrain)
                Flux.trainmode!(model)
                if epoch%(epochNum÷5)==1
                    #ccTrain = cor(model_cpu(X)[:],Y[:])
                    @show epoch
                    @show lossTrain
                    #@show ccTrain
                end
            end
        end
        =#


        evalcb = Flux.throttle(10) do
          # 因为有BatchNorm层，测试时使用testmode
          Flux.testmode!(model)
          model_cpu = cpu(model)
          lossTrain = mse(model_cpu(X),Y)
          push!(trainCost,lossTrain)
          Flux.trainmode!(model)
        end
        Flux.@epochs epochNum begin
            Flux.train!(calCost,ps,trainData,opt,cb=evalcb)
        end


        Flux.testmode!(model)
        model_cpu = cpu(model)

        local losscv = mse(model_cpu(Xcv),Ycv)*ny
        #local cccv = cor(model_cpu(Xcv)[:],Ycv[:])
        #local cctrain = cor(model_cpu(X)[:],Y[:])
        push!(cvCosts,losscv)
        #push!(cvCCs,cccv)
        #push!(trainCCs,cctrain)
        Flux.trainmode!(model)
        push!(Costs,trainCost)
    end

    #Costs,cvCosts,cvCCs,trainCCs
    Costs,cvCosts
end
#=
SWTYPE = "ST"
figIndex = string(3)
theParameterName = "α"
#theParameters = [Float32(10^-1.),]
theParameters = 10 .^(Float32(1)*rand(Float32,5).-Float32(3))
#theParameterName = "λ"

epochNum = Int32(2001)
#testCosts,cvCosts,cvCCs,trainCCs = testModels(
testCosts,cvCosts = testModels(
theParameterName,
theParameters,
epochNum,
X2,
Y2,
X2cv,
Y2cv;
buildmymodel = buildModel2,
α=Float32(0.006),
λ=Float32(1),
)


Plots.plot(testCosts,
label=round.(theParameters,digits=5)',
xlabel="epoch",
ylabel="mse of train set",
ylims=(0.35,0.45),
ls=:auto)
savefig("figure\\"*
DataName*
"\\"*
SWTYPE*
"\\J"*
theParameterName*
figIndex*
".png")

if theParameterName == "λ"
    Plots.scatter(
        log10.(theParameters),
        cvCosts,
        label = "cvcost",
        xlabel = theParameterName,
    )
    savefig(
        "figure\\" *
        DataName *
        "\\" *
        SWTYPE *
        "\\cvcost" *
        theParameterName *
        figIndex *
        ".png",
    )
#=
    Plots.scatter(
        log10.(theParameters),
        cvCCs,
        label = "cvCC",
        xlabel = theParameterName,
    )
    savefig(
        "figure\\" *
        DataName *
        "\\" *
        SWTYPE *
        "\\cvcc" *
        theParameterName *
        figIndex *
        ".png",
    )
    =#
end
=#

#####################Test##########################

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
            #ccTrain = cor(model_cpu(X)[:], Y[:])
            Flux.trainmode!(model)
            @show epoch
            @show lossTrain
            #@show ccTrain
        end
    end

    Flux.testmode!(model)
    model_cpu = cpu(model)
    local losscv = mse(model_cpu(Xcv), Ycv)*ny
    #local cccv = cor(model_cpu(Xcv)[:], Ycv[:])
    local losstrain = mse(model_cpu(X), Y)*ny
    #local cctrain = cor(model_cpu(X)[:], Y[:])
    #model, cccv, losscv
    model,losscv
end
#=
epochNum = Int32(1001)

themodel1,thelosscv1 = trainOneModel(epochNum,
X1,
Y1,
X1cv,
Y1cv;
buildmymodel = buildModel2,
α=Float32(0.003),
λ=Float32(0.1),
)
themodel2,thelosscv2 = trainOneModel(Int32(2001),
X2,
Y2,
X2cv,
Y2cv;
buildmymodel = buildModel2,
α=Float32(0.006),
λ=Float32(1.),
)
themodel3,thelosscv3 = trainOneModel(Int32(2001),
X3,
Y3,
X3cv,
Y3cv;
buildmymodel = buildModel2,
α=Float32(0.035),
λ=Float32(0.1),
)

#@load "data\\tot\\models.jld2" themodel1 themodel2 themodel3
#device =
Ŷ1 = cpu(themodel1)(X1)
Ŷ1cv = cpu(themodel1)(X1cv)
Ŷ1test = cpu(themodel1)(X1test)
Ŷ2 = cpu(themodel2)(X2)
Ŷ2cv = cpu(themodel2)(X2cv)
Ŷ2test = cpu(themodel2)(X2test)
Ŷ3 = cpu(themodel3)(X3)
Ŷ3cv = cpu(themodel3)(X3cv)
Ŷ3test = cpu(themodel3)(X3test)

CCtrain1 = cor(Y1,Ŷ1,dims=2)
CCdev1 = cor(Y1cv,Ŷ1cv,dims=2)
CCtest1 = cor(Y1test,Ŷ1test,dims=2)
CCtrain2 = cor(Y2,Ŷ2,dims=2)
CCdev2 = cor(Y2cv,Ŷ2cv,dims=2)
CCtest2 = cor(Y2test,Ŷ2test,dims=2)
CCtrain3 = cor(Y3,Ŷ3,dims=2)
CCdev3 = cor(Y3cv,Ŷ3cv,dims=2)
CCtest3 = cor(Y3test,Ŷ3test,dims=2)


Ys = [Y1 Y2 Y3]
Ycvs = [Y1cv Y2cv Y3cv]
Ytests = [Y1test Y2test Y3test]
Ŷs = [Ŷ1 Ŷ2 Ŷ3]
Ŷcvs = [Ŷ1cv Ŷ2cv Ŷ3cv]
Ŷtests = [Ŷ1test Ŷ2test Ŷ3test]

cctrain = cor(Ys,Ŷs,dims=2)
ccdev = cor(Ycvs,Ŷcvs,dims=2)
cctest = cor(Ytests,Ŷtests,dims=2)

CCtrains = [CCtrain1,CCtrain2,CCtrain3]
CCdevs = [CCdev1,CCdev2,CCdev3]
CCtests = [CCtest1,CCtest2,CCtest3]
SWtypes = ["CH" "ST" "CME"]
SWtypenames = ["CH" "Streamer" "CME"]
Datanames = ["FetoO" "O7to6" "C6to5" "C6to4"]
=#

#=
for i in eachindex(SWtypes)
    theSWtype = SWtypes[i]
    theSWtypename = SWtypenames[i]
    theY = Ys[i]
    theYcv = Ycvs[i]
    theYtest = Ytests[i]
    theŶ = Ŷs[i]
    theŶcv = Ŷcvs[i]
    theŶtest = Ŷtests[i]
    for j in eachindex(Datanames)
        theDataName = Datanames[j]
        Plots.scatter(
            theY[j,:],
            theŶ[j,:],
            xlabel = "real",
            ylabel = "predict",
            #title = "train Set CC="*string(round(cctrain,digits=4)),
            label="CC="*string(round(CCtrains[i][j,j],digits=4)),
            aspect_ratio = :equal,
            ms = 1,
            xlims=(minimum(theY[j,:]),2*mean(theY[j,:])-minimum(theY[j,:])),
            ylims=(minimum(theY[j,:]),2*mean(theY[j,:])-minimum(theY[j,:])),
            )
            plot!(x->x,
            label=nothing,
            ls=:dash)
        savefig("figure\\tot\\scatter\\"*theDataName*theSWtype*"Train.png")

        Plots.scatter(
            theYcv[j,:],
            theŶcv[j,:],
            xlabel = "real",
            ylabel = "predict",
            #title = "train Set CC="*string(round(cctrain,digits=4)),
            label="CC="*string(round(CCdevs[i][j,j],digits=4)),
            aspect_ratio = :equal,
            ms = 1,
            xlims=(minimum(theYcv[j,:]),2*mean(theYcv[j,:])-minimum(theYcv[j,:])),
            ylims=(minimum(theYcv[j,:]),2*mean(theYcv[j,:])-minimum(theYcv[j,:])),
            )
            plot!(x->x,
            label=nothing,
            ls=:dash)
        savefig("figure\\tot\\scatter\\"*theDataName*theSWtype*"Dev.png")

        Plots.scatter(
            theYtest[j,:],
            theŶtest[j,:],
            xlabel = "real",
            ylabel = "predict",
            #title = "train Set CC="*string(round(cctrain,digits=4)),
            label="CC="*string(round(CCtests[i][j,j],digits=4)),
            aspect_ratio = :equal,
            ms = 1,
            xlims=(minimum(theYtest[j,:]),2*mean(theYtest[j,:])-minimum(theYtest[j,:])),
            ylims=(minimum(theYtest[j,:]),2*mean(theYtest[j,:])-minimum(theYtest[j,:])),
            )
            plot!(x->x,
            label=nothing,
            ls=:dash)
        savefig("figure\\tot\\scatter\\"*theDataName*theSWtype*"Test.png")

        p = sortperm(theY[j,:])
        scatter(
        [ theŶ[j,p] theY[j,p] ],
        label=["predict" "real"],
        #title=outputNames[i]*" Train set",
        ms=0.1,
        color=[:green :red ],
        markerstrokewidth=0,
        xlabel="data counts",
        )
        savefig("figure\\tot\\scatter\\"*theDataName*theSWtype*"Train2.png")
        p = sortperm(theYcv[j,:])
        scatter(
        [ theŶcv[j,p] theYcv[j,p] ],
        label=["predict" "real"],
        #title=outputNames[i]*" Dev set",
        ms=0.1,
        color=[:green :red ],
        markerstrokewidth=0,
        xlabel="data counts",
        )
        savefig("figure\\tot\\scatter\\"*theDataName*theSWtype*"Dev2.png")
        p = sortperm(theYtest[j,:])
        scatter(
        [ theŶtest[j,p] theYtest[j,p] ],
        label=["predict" "real"],
        #title=outputNames[i]*" test set",
        ms=0.1,
        color=[:green :red ],
        markerstrokewidth=0,
        xlabel="data counts",
        )
        savefig("figure\\tot\\scatter\\"*theDataName*theSWtype*"Test2.png")
    end
end
=#

#=  ##########################################
for j in eachindex(Datanames)
    theDataName = Datanames[j]
    Plots.scatter(
        Ys[j,:],
        Ŷs[j,:],
        xlabel = "real",
        ylabel = "predict",
        #title = "train Set CC="*string(round(cctrain,digits=4)),
        label="CC="*string(round(cctrain[j,j],digits=4)),
        aspect_ratio = :equal,
        ms = 1,
        xlims=(minimum(Y[j,:]),2.5*mean(Y[j,:])),
        ylims=(minimum(Y[j,:]),2.5*mean(Y[j,:])),
        )
        plot!(x->x,
        label=nothing,
        ls=:dash)
    savefig("figure\\tot\\scatter\\tot\\"*theDataName*"Train.png")

    Plots.scatter(
        Ycvs[j,:],
        Ŷcvs[j,:],
        xlabel = "real",
        ylabel = "predict",
        #title = "train Set CC="*string(round(cctrain,digits=4)),
        label="CC="*string(round(ccdev[j,j],digits=4)),
        aspect_ratio = :equal,
        ms = 1,
        xlims=(minimum(Ycvs[j,:]),2.5*mean(Ycvs[j,:])),
        ylims=(minimum(Ycvs[j,:]),2.5*mean(Ycvs[j,:])),
        )
        plot!(x->x,
        label=nothing,
        ls=:dash)
    savefig("figure\\tot\\scatter\\tot\\"*theDataName*"Dev.png")

    Plots.scatter(
        Ytests[j,:],
        Ŷtests[j,:],
        xlabel = "real",
        ylabel = "predict",
        #title = "train Set CC="*string(round(cctrain,digits=4)),
        label="CC="*string(round(cctest[j,j],digits=4)),
        aspect_ratio = :equal,
        ms = 1,
        xlims=(minimum(Ytests[j,:]),2.5*mean(Ytests[j,:])),
        ylims=(minimum(Ytests[j,:]),2.5*mean(Ytests[j,:])),
        )
        plot!(x->x,
        label=nothing,
        ls=:dash)
    savefig("figure\\tot\\scatter\\tot\\"*theDataName*"Test.png")

    p = sortperm(Ys[j,:])
    scatter(
    [ Ŷs[j,p] Ys[j,p] ],
    label=["predict" "real"],
    #title=outputNames[i]*" Train set",
    ms=0.1,
    color=[:green :red ],
    markerstrokewidth=0,
    xlabel="data counts",
    )
    savefig("figure\\tot\\scatter\\tot\\"*theDataName*"Train2.png")
    p = sortperm(Ycvs[j,:])
    scatter(
    [ Ŷcvs[j,p] Ycvs[j,p] ],
    label=["predict" "real"],
    #title=outputNames[i]*" Dev set",
    ms=0.1,
    color=[:green :red ],
    markerstrokewidth=0,
    xlabel="data counts",
    )
    savefig("figure\\tot\\scatter\\tot\\"*theDataName*"Dev2.png")
    p = sortperm(Ytests[j,:])
    scatter(
    [ Ŷtests[j,p] Ytests[j,p] ],
    label=["predict" "real"],
    #title=outputNames[i]*" test set",
    ms=0.1,
    color=[:green :red ],
    markerstrokewidth=0,
    xlabel="data counts",
    )
    savefig("figure\\tot\\scatter\\tot\\"*theDataName*"Test2.png")
end
=#  #######################################



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
    local ny = size(y,1)

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
#=
epochNum = Int32(801)
themodel,thecccv = chooseBestModel(epochNum,
XCls,
YCls,
XcvCls,
YcvCls;
buildmymodel = buildModel,
α=Float32(0.004),
λ=Float32(0.056),
)
=#

function getModels(X,Y,myClusteringResult,DatacvAssignments,k,αs,λs;epochNum=Int32(801))

    models = []
    for clusterType in 1:k
        println()
        local α=αs[clusterType]
        local λ=λs[clusterType]
        @show clusterType

        XCls = X[:,findall(myClusteringResult.assignments.==clusterType)]
        YCls = Y[:,findall(myClusteringResult.assignments.==clusterType)]
        XcvCls = Xcv[:,findall(DatacvAssignments.==clusterType)]
        YcvCls = Ycv[:,findall(DatacvAssignments.==clusterType)]

        themodel,thecccv = chooseBestModel(epochNum,
        XCls,
        YCls,
        XcvCls,
        YcvCls;
        buildmymodel = buildModel,
        α=α,
        λ=λ,
        )
        @show thecccv
        push!(models,themodel)
    end
    models
end
#=
k=3
αs=Float32.([0.004,0.004,0.006])
λs=Float32.([0.01,0.1,0.2])
models = getModels(X,Y,myClusteringResult,DatacvAssignments,k,αs,λs;epochNum=Int32(801))

DatatestAssignments = predict(Ztest,myClusteringResult.centers)
=#

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

################# 这部分画结果的散点图########################
#=
αs=Float32.([0.0004,0.005,0.006,0.005,0.005,0.0025])
λs=Float32.([1.,0.4,0.1,0.05,1e-4,0.9])
models,devcvs = getTotModel(X,Y,Xcv,Ycv,αs,λs;epochNum=Int32(1001))
=#
#@save "data\\models.jld2" models devcvs
# @load "data\\models.jld2" models
# outputNames = ["FetoO" "O7toO6" "C6toC5" "C6toC4" "nHe2" "vHe2"]
# outputTitles = ["Fe/O" "O7/O6" "C6/C5" "C6/C4" "Nα" "Vα"]
# panelIdxs = ["a" "b" "c" "d" "e" "f"]
#
# figs = []
# for i in eachindex(models)
#     model = models[i]
#     Ŷ = model(X)
#     Ŷcv = model(Xcv)
#     Ŷtest = model(Xtest)
#     cctrain = round(cor(Ŷ[:],Y[i,:]),digits=3)
#     cccv = round(cor(Ŷcv[:],Ycv[i,:]),digits=3)
#     cctest = round(cor(Ŷtest[:],Ytest[i,:]),digits=3)
#     l1 = @layout [a b c]
#     s1 =
#     # scatter(
#     histogram2d(
#     Y[i,:],
#     Ŷ[:],
#     legend=false,
#     xlabel=i==6 ? "real" : "",
#     ylabel=i==6 ? "predict"*" cc="*string(cctrain) : "cc="*string(cctrain),
#     title="("*panelIdxs[i]*"1) "*outputTitles[i]*" (train set)",
#     titlefont=25,
#     guidefont=22,
#     tickfont=14,
#     # label="cc="*string(cctrain),
#     aspect_ratio = :equal,
#     ms = 1,
#     xlims=(minimum(Y[i,:]),2.5*mean(Y[i,:])),
#     ylims=(minimum(Y[i,:]),2.5*mean(Y[i,:])),
#     colorbar=false,
#     )
#     if i==1
#         xticks!([0.05;0.15;0.25;0.35])
#     end
#     plot!(x->x,
#     label=nothing,
#     ls=:dash)
#     #savefig("figure\\scatters\\"*outputNames[i]*"Train.png")
#     s2 =
#     # scatter(
#     histogram2d(
#     Ycv[i,:],
#     Ŷcv[:],
#     legend=false,
#     xlabel=i==6 ? "real" : "",
#     ylabel=i==6 ? "predict"*" cc="*string(cccv) : "cc="*string(cccv),
#     title="("*panelIdxs[i]*"2) "*outputTitles[i]*" (dev set)",
#     titlefont=25,
#     guidefont=22,
#     tickfont=14,
#     # label="cc="*string(cccv),
#     aspect_ratio = :equal,
#     ms = 1,
#     xlims=(minimum(Y[i,:]),2.5*mean(Y[i,:])),
#     ylims=(minimum(Y[i,:]),2.5*mean(Y[i,:])),
#     colorbar=false,
#     )
#     if i==1
#         xticks!([0.05;0.15;0.25;0.35])
#     end
#     plot!(x->x,
#     label=nothing,
#     ls=:dash)
#     #savefig("figure\\scatters\\"*outputNames[i]*"Dev.png")
#     s3 =
#     # scatter(
#     histogram2d(
#     Ytest[i,:],
#     Ŷtest[:],
#     legend=false,
#     xlabel=i==6 ? "real" : "",
#     ylabel=i==6 ? "predict"*" cc="*string(cctest) : "cc="*string(cctest),
#     title="("*panelIdxs[i]*"3) "*outputTitles[i]*" (test set)",
#     titlefont=25,
#     guidefont=22,
#     tickfont=14,
#     # label="cc="*string(cctest),
#     aspect_ratio = :equal,
#     ms = 1,
#     xlims=(minimum(Y[i,:]),2.5*mean(Y[i,:])),
#     ylims=(minimum(Y[i,:]),2.5*mean(Y[i,:])),
#     colorbar=false,
#     )
#     if i==1
#         xticks!([0.05;0.15;0.25;0.35])
#     end
#     plot!(x->x,
#     label=nothing,
#     ls=:dash)
#     #savefig("figure\\scatters\\"*outputNames[i]*"Test.png")
#     push!(figs,s1)
#     push!(figs,s2)
#     push!(figs,s3)
#     #=
#     p = sortperm(Y[i,:])
#     scatter([ Ŷ[p] Y[i,p] ],
#     label=["predict" "real"],
#     title=outputNames[i]*" Train set",
#     ms=0.1,
#     color=[:green :red ],
#     markerstrokewidth=0,
#     xlabel="data counts",
#     )
#     savefig("figure\\scatters\\"*outputNames[i]*"train2.png")
#     p = sortperm(Ycv[i,:])
#     scatter([ Ŷcv[p] Ycv[i,p] ],
#     label=["predict" "real"],
#     title=outputNames[i]*" Dev set",
#     ms=0.1,
#     color=[:green :red ],
#     markerstrokewidth=0,
#     xlabel="data counts",
#     )
#     savefig("figure\\scatters\\"*outputNames[i]*"dev2.png")
#     p = sortperm(Ytest[i,:])
#     scatter([ Ŷtest[p] Ytest[i,p] ],
#     label=["predict" "real"],
#     title=outputNames[i]*" test set",
#     ms=0.1,
#     color=[:green :red ],
#     markerstrokewidth=0,
#     xlabel="data counts",
#     )
#     savefig("figure\\scatters\\"*outputNames[i]*"test2.png")
#     =#
# end
# l = @layout grid(6,3)
# plot(figs...,
# layout = l,
# size=(1800,2900),
# )
# savefig("figure\\paper\\scatter\\toth2.png")

function contribution(X, Y, Xcv, Ycv, n1s::Vector{Int},α,λ)
    nx = size(X, 1)

    CostTrainData = zeros(nx,length(n1s))
    CCTrainData = copy(CostTrainData)
    CostCVData = copy(CostTrainData)
    CCCVData = copy(CostTrainData)
    Xindexs = collect(1:nx)
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
# MSETrain,MSEDev,CCTrain,CCDev = contributionTol(X,Y,Xcv,Ycv,αs,λs,n1s;epochNum=Int32(1001))
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
savefig("figure\\paper\\contributions\\all\\allDev.png")





function calYhat(DataAssignments,Data,k,models)
    Yhat = zeros(Float32,(1,length(DataAssignments)))
    for clusterType in 1:k
        local model = models[clusterType]
        Yhat[findall(DataAssignments.==clusterType)] = model(Data[:,findall(DataAssignments.==clusterType)])
    end
    Yhat
end

function testCandR!(k,Yhat,Y,DataAssignments)
    for clusterType in 1:k
        @show clusterType
        theCC = cor(Yhat[findall(DataAssignments.==clusterType)],
        Y[findall(DataAssignments.==clusterType)])
        @show theCC
    end
    Cc = cor(Y[:],Yhat[:])
    @show Cc
    nothing
end

#=
Ŷcv = calYhat(DatacvAssignments,Xcv,k,models)
testCandR!(k,Ŷcv,Ycv,DatacvAssignments)
Ŷtest = calYhat(DatatestAssignments,Xtest,k,models)
testCandR!(k,Ŷtest,Ytest,DatatestAssignments)

=#

##############softmax
#=
#=
"""
用于softmax的模型
"""
function buildModel3(nx::Int64,λ::Float32,minibatchsize::Int64)
    local model = Chain(Dense(nx,3*nx),
    BatchNorm(3*nx,leakyrelu),
    Dense(3*nx,nx),
    BatchNorm(nx,leakyrelu),
    Dense(nx,4)) |> gpu
    local regularParams = Flux.params(model)
    delete!(regularParams,model[1].b)
    delete!(regularParams,model[3].b)
    delete!(regularParams,model[5].b)
    delete!(regularParams,model[2].γ)
    delete!(regularParams,model[2].β)
    delete!(regularParams,model[4].γ)
    delete!(regularParams,model[4].β)
    function calCost(X,Y)
        sqnorm(x) = sum(abs2,x)
        Flux.Losses.logitcrossentropy(model(X),Y) + λ*sum(sqnorm,regularParams)/minibatchsize
    end
    ps = Flux.params(model)
    delete!(ps,model[1].b)
    delete!(ps,model[3].b)
    model,calCost,ps
end
=#

"""
记得改：
默认的学习率、正则化系数
选择参数是哪个
选择模型是否输入nh
不同模型参数类型可能要改
"""
function testModels(theParameters::Vector,epochNum::Int32,X::Array{Float32,2},Y::Array{Bool,2},Xcv::Array{Float32,2},Ycv::Array{Bool,2};
    α=Float32(0.005),minibatchsize=1024,λ=Float32(0.005),buildmymodel = buildModel3)
    local Costs = []
    local trainACCs = []
    local cvCosts = []
    local cvACCs = []
    for theParameter in theParameters
        #minibatchsize = theParameter
        λ = theParameter
        #α = theParameter
        #nh = Int32(theParameter)
        println()
        #@show theParameter
        @show α
        @show minibatchsize
        @show λ

        opt = ADAM(α)
        nx = Int32(size(X,1))

        #model,calCost,ps = buildmymodel(nx,nh,λ,minibatchsize)
        model,calCost,ps = buildmymodel(nx,λ,minibatchsize)

        local trainCost = []
        trainData = Flux.Data.DataLoader(
        (gpu(X), gpu(Y)),
        batchsize=minibatchsize,
        shuffle=true)
        for epoch in 1:epochNum
            Flux.train!(calCost,ps,trainData,opt)
            if epoch%(epochNum÷100)==1
                Flux.testmode!(model)
                model_cpu = cpu(model)
                lossTrain = Flux.Losses.logitcrossentropy(model_cpu(X),Y)
                push!(trainCost,lossTrain)
                Flux.trainmode!(model)
                if epoch%(epochNum÷10)==1
                    accTrain = mean(Flux.onecold(model_cpu(X),[0,1,2,3]).==SWflag[:])
                    @show epoch
                    @show lossTrain
                    @show accTrain
                end
            end
        end

        #=
        evalcb = Flux.throttle(1) do
          # 因为有BatchNorm层，测试时使用testmode
          Flux.testmode!(model)
          model_cpu = cpu(model)
          lossTrain = mse(model_cpu(X),Y)
          push!(trainCost,lossTrain)
          Flux.trainmode!(model)
        end
        Flux.@epochs epochNum begin
            Flux.train!(calCost,ps,trainData,opt,cb=evalcb)
        end
        =#

        Flux.testmode!(model)
        model_cpu = cpu(model)

        function calPrecisionRecall(swtype::Int)
            precision = mean(
            SWflag[
            findall(Flux.onecold(model_cpu(X),[0,1,2,3]).==swtype)
            ].==swtype)
            recall = mean(
            Flux.onecold(model_cpu(X),[0,1,2,3])[
            findall(SWflag'.==swtype)
            ].==swtype)
            precision,recall
        end

        function calPRs(SWTYPEs)
            for i in eachindex(SWTYPEs)
                @show SWTYPEs[i]
                thep,theR = calPrecisionRecall(i-1)
                @show thep
                @show theR
            end
            nothing
        end

        calPRs(["ST","CH","CME"])
        local losscv = mse(model_cpu(Xcv),Ycv)
        local acccv = mean(Flux.onecold(model_cpu(Xcv),[0,1,2,3]).==SWflagcv[:])
        local acctrain = mean(Flux.onecold(model_cpu(X),[0,1,2,3]).==SWflag[:])
        push!(cvCosts,losscv)
        push!(cvACCs,acccv)
        push!(trainACCs,acctrain)
        Flux.trainmode!(model)
        push!(Costs,trainCost)
    end
    #Flux.@functor Dense (W,b)
    Costs,cvCosts,cvACCs,trainACCs
end
#=
SWTYPE = "tot"
#theParameterName = "α"
theParameters = 10 .^(Float32(5)*rand(Float32,5).-Float32(8))
theParameterName = "λ"
#theParameters = 10 .^(Float32(1.5)*rand(Float32,5).-Float32(.5))
#theParameters = Float32(0.01) .+ Float32(0.24)*rand(Float32,20)
#theParameters = Float32(0.01) .+ Float32(0.1)*rand(Float32,6)
epochNum = Int32(601)
testCosts,cvCosts,cvACCs,trainACCs = testModels(
theParameters,
epochNum,
X,
Y,
Xcv,
Ycv;
buildmymodel = buildModel3)

figIndex = string(1)
Plots.plot(testCosts,
label=round.(theParameters,digits=5)',
xlabel="epoch",
ylabel="mse of train set",
ylims=(0.48,0.6),
ls=:auto)
savefig("figure\\"*DataName*"\\"*SWTYPE*"\\J"*theParameterName*figIndex*".png")


Plots.scatter(log10.(theParameters),
cvCosts,
label="cvcost",
xlabel=theParameterName,
)
savefig("figure\\"*DataName*"\\"*SWTYPE*"\\cvcost"*theParameterName*figIndex*".png")

Plots.scatter(log10.(theParameters),
cvACCs,
label="cvACC",
xlabel=theParameterName,
)
savefig("figure\\"*DataName*"\\"*SWTYPE*"\\cvcc"*theParameterName*figIndex*".png")
=#

#train a model
#=
SWTYPE = "ST"
theX = X2
theY = Y2
theXcv = X2cv
theYcv = Y2cv
λ = Float32(4)
nx = Int(9)
α = 0.0042
epochNum = 2001
model,calCost,ps = buildTestModel1(nx,3,λ,1024)
opt = ADAM(α)
trainCost = []
trainData = Flux.Data.DataLoader(
(gpu(theX), gpu(theY)),
batchsize=1024,
shuffle=true)
for epoch in 1:epochNum
    Flux.train!(calCost,ps,trainData,opt)
    if epoch%(epochNum÷100)==1
        Flux.testmode!(model)
        local model_cpu = cpu(model)
        lossTrain = mse(model_cpu(theX),theY)
        push!(trainCost,lossTrain)
        Flux.trainmode!(model)
        if epoch%(epochNum÷10)==1
            @show epoch
            @show lossTrain
        end
    end
end
Flux.testmode!(model)
model_cpu = cpu(model)
losscv = mse(model_cpu(theXcv),theYcv)
cccv = cor(model_cpu(theXcv)[:],theYcv[:])

cctrain = cor(model_cpu(theX)[:],theY[:])
@show losscv
@show cccv
@show cctrain
Flux.trainmode!(model)
plot(trainCost)
savefig("figure\\test\\J.png")

Plots.scatter(
    model_cpu(theX)[:],
    theY[:],
    xlabel = "predict",
    ylabel = "real",
    title = "train Set CC="*string(round(cctrain,digits=4)),
    label=DataName*" "*SWTYPE,
    aspect_ratio = :equal,
    ms = 1,
    #xlims=(0,1),
    #ylims=(0,1),
)
savefig("figure\\"*DataName*"\\"*SWTYPE*"\\scatterTrain.png")

Plots.scatter(
    model_cpu(theXcv)[:],
    theYcv[:],
    xlabel = "predict",
    ylabel = "real",
    title = "dev Set CC="*string(round(cccv,digits=4)),
    label=DataName*" "*SWTYPE,
    aspect_ratio = :equal,
    ms = 1,
    #xlims=(0,1),
    #ylims=(0,1),
)
savefig("figure\\"*DataName*"\\"*SWTYPE*"\\scatterDev.png")
=#

#=
"""
可以改：
训练次数epoches
"""
function contributionMinus(X,Y,Xcv,Ycv,nhs)
    nx = size(X,1)
    Xindexs = collect(1:nx)
    local CVCOST = []
    local CVCC = []
    local TRAINCC = []
    for xindex in 1:nx
        @show xindex
        thexindexs = filter(x->x!=xindex,Xindexs)
        theX = X[thexindexs,:]
        theXcv = Xcv[thexindexs,:]
        ~,cvCosts,cvCCs,trainCCs = testModels(
        nhs,
        Int32(3001),
        theX,
        Y,
        theXcv,
        Ycv;
        buildmymodel = buildTestModel1)
        push!(CVCOST,cvCosts)
        push!(CVCC,cvCCs)
        push!(TRAINCC,trainCCs)
    end
    ~,cvCosts,cvCCs,trainCCs = testModels(
    nhs,
    Int32(3001),
    X,
    Y,
    Xcv,
    Ycv;
    buildmymodel = buildTestModel1)
    push!(CVCOST,cvCosts)
    push!(CVCC,cvCCs)
    push!(TRAINCC,trainCCs)
    CVCOST,CVCC,TRAINCC
end
SWTYPE = "CH"
nhs = [16,32,60,100]
CVCOST,CVCC,TRAINCC = contributionMinus(X1,Y1,X1cv,Y1cv,nhs)
@save "data\\"*DataName*"\\"*SWTYPE*"\\contributionMinus.jld2" CVCOST CVCC TRAINCC
l = @layout [a b]
labels = ["B" "np" "Vp" "Vthp" "δBrms" "δVrms" "σc" "SSN" "F10.7" "All"]
Pcost = Plots.plot(nhs,CVCOST[:];label=labels,marker=5,
xlabel="Node number",ylabel="MSE",title=DataName,
legend=:bottom,
shape=:auto,
#ylim=(0.002,0.0023),
)
Pcc = Plots.plot(nhs,CVCC[:];label=labels,marker=5,
xlabel="Node number",ylabel="correlation coefficient",
title=DataName,legend=:bottom,shape=:auto)
Plots.plot(Pcc,Pcost,layout=l)
savefig("figure\\"*DataName*"\\"*SWTYPE*"\\MinusCostandCCCV.png")
=#

#=
function contribution(X,Y,Xcv,Ycv,nhs)
    nx = size(X,1)
    local CVCOST = []
    local CVCC = []
    local TRAINCC = []
    for xindex in 1:nx
        @show xindex
        theX = reshape(X[xindex,:],(1,:))
        theXcv = reshape(Xcv[xindex,:],(1,:))
        ~,cvCosts,cvCCs,trainCCs = testModels(
        nhs,
        Int32(1001),
        theX,
        Y,
        theXcv,
        Ycv;
        buildmymodel = buildTestModel1)
        push!(CVCOST,cvCosts)
        push!(CVCC,cvCCs)
        push!(TRAINCC,trainCCs)
    end
    CVCOST,CVCC,TRAINCC
end
SWTYPE = "tot"
#nhs = [100,]
nhs = [20,40,70,100]
CVCOST,CVCC,TRAINCC = contribution(X,Y,Xcv,Ycv,nhs)
@save "data\\"*DataName*"\\"*SWTYPE*"\\contribution.jld2" CVCOST CVCC TRAINCC

l = @layout [a b]
labels = ["B" "np" "Vp" "Vthp" "δBrms" "δVrms" "σc" "SSN" "F10.7"]
Pcost = Plots.plot(nhs,CVCOST[:];label=labels,marker=5,
xlabel="Node number",ylabel="MSE",title=DataName,
legend=:bottom,
shape=:auto,
#ylim=(0.035,0.061),
)
Pcc = Plots.plot(nhs,CVCC[:];label=labels,marker=5,
xlabel="Node number",ylabel="correlation coefficient",
#ylims=(0.,0.52),
title=DataName,legend=:bottom,shape=:auto)
Plots.plot(Pcc,Pcost,layout=l)
savefig("figure\\"*DataName*"\\"*SWTYPE*"\\contribution.png")
=#
=#
