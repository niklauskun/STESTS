using STESTS, JuMP, Gurobi, CSV, DataFrames, Statistics, SMTPClient

Year = 2022
# Read data from .jld2 file 
params = STESTS.read_jld2("./data/ADS2032_5GWBES_BS_" * "$Year" * ".jld2")
strategic = false
RandomModel = false
RandomSeed = 1
ratio = 1.0
RM = 0.03
VOLL = 9000.0
NDay = 2
UCHorizon = Int(25) # optimization horizon for unit commitment model, 24 hours for WECC data, 4 hours for 3-bus test data
EDHorizon = Int(1) # optimization horizon for economic dispatch model, 1 without look-ahead, 12 with 1-hour look-ahead
EDSteps = Int(12) # number of 5-min intervals in a hour
ESSeg = Int(1)
ESMC = 20.0
BAWindow = Int(0) # bid-ahead window (number of 5-min intervals, 12-1hr, 48-4hr)
PriceCap = repeat(
    repeat(
        (range(220, stop = 1000, length = 40))',
        outer = (size(params.UCL, 2), 1),
    ),
    outer = (1, 1, EDHorizon),
)
FuelAdjustment = 1.2
ErrorAdjustment = 0.25
LoadAdjustment = 1.0

output_folder =
    "output/Strategic/" *
    "$Year" *
    "/UC" *
    "$UCHorizon" *
    "ED" *
    "$EDHorizon" *
    "_Strategic_" *
    "$strategic" *
    "_ratio" *
    "$ratio" *
    "_Seg" *
    "$ESSeg" *
    "_BAW" *
    "$BAWindow" *
    "_MC" *
    "$ESMC" *
    "_" *
    "$RandomSeed" *
    "test2"
mkpath(output_folder)

# model_filenames = [
#     "models/BAW" *
#     "$BAWindow" *
#     "EDH" *
#     "$EDHorizon" *
#     "MC" *
#     "$ESMC" *
#     "/Region1/4hrmodel1_5Seg.jld2",
# ]
model_base_folder =
    "models/BAW" * "$BAWindow" * "EDH" * "$EDHorizon" * "MC" * "$ESMC"

# Update strategic storage scale base on set ratio
storagebidmodels = []
if strategic
    mkpath(output_folder * "/Strategic")
    mkpath(output_folder * "/NStrategic")
    if ratio == 1.0
        for i in eachindex(params.Eeta)
            if params.Eeta[i] == 0.8
                params.EStrategic[i] = 0
            elseif params.Eeta[i] == 0.9
                params.EStrategic[i] = 1
            else
                # Handle unexpected case, if necessary
                println(
                    "Unexpected value in params.Eeta at index $i: ",
                    params.Eeta[i],
                )
            end
        end
    elseif ratio == 0.0
        println("No AI-Powered BES.")
    else
        STESTS.update_battery_storage!(params, ratio, output_folder)
    end
    # bidmodels = STESTS.loadbidmodels(model_filenames)
    bidmodels = STESTS.loadbidmodels(model_base_folder)
    storagebidmodels = STESTS.assign_models_to_storages(
        params,
        bidmodels,
        size(params.storagemap, 1),
        output_folder,
        RandomModel = RandomModel,
        RandomSeed = RandomSeed,
    )
end

DADBids = repeat(params.ESDABids[:, 1]', size(params.storagemap, 1), 1)
DACBids = repeat(params.ESDABids[:, 2]', size(params.storagemap, 1), 1)
RTDBids = repeat(params.ESRTBids[:, 1]', size(params.storagemap, 1), 1)
RTCBids = repeat(params.ESRTBids[:, 2]', size(params.storagemap, 1), 1)

# Formulate unit commitment model
ucmodel = STESTS.unitcommitment(
    params,
    Horizon = UCHorizon, # optimization horizon for unit commitment model, 24 hours for WECC data, 4 hours for 3-bus test data
    VOLL = VOLL, # value of lost load, $/MWh
    RM = RM, # reserve margin
    FuelAdjustment = FuelAdjustment,
)

# Edit unit commitment model here
# set optimizer, set add_bridges = false if model is supported by solver
set_optimizer(ucmodel, Gurobi.Optimizer, add_bridges = false)
set_optimizer_attribute(ucmodel, "OutputFlag", 0)
# # modify objective function
# @objective(ucmodel, Min, 0.0)
# # modify or add constraints
# @constraint(ucmodel, 0.0 <= ucmodel[:P][1,1] <= 0.0)

ucpmodel = STESTS.unitcommitmentprice(
    params,
    Horizon = UCHorizon, # optimization horizon for unit commitment model, 24 hours for WECC data, 4 hours for 3-bus test data
    VOLL = VOLL, # value of lost load, $/MWh
    RM = RM, # reserve margin
    FuelAdjustment = FuelAdjustment,
)

# Edit unit commitment model here
# set optimizer, set add_bridges = false if model is supported by solver
set_optimizer(ucpmodel, Gurobi.Optimizer, add_bridges = false)
set_optimizer_attribute(ucpmodel, "OutputFlag", 0)
# # modify objective function
# @objective(ucpmodel, Min, 0.0)
# # modify or add constraints
# @constraint(ucpmodel, 0.0 <= ucpmodel[:P][1,1] <= 0.0)

#  Formulate economic dispatch model
edmodel = STESTS.economicdispatch(
    params,
    PriceCap, # value of lost load, $/MWh
    ESSeg = ESSeg,
    Horizon = EDHorizon,
    Steps = EDSteps, # optimization horizon for unit commitment model, 24 hours for WECC data, 4 hours for 3-bus test data
    FuelAdjustment = FuelAdjustment,
)

# Edit economic dispatch model here
# set optimizer, set add_bridges = false if model is supported by solver
set_optimizer(edmodel, Gurobi.Optimizer, add_bridges = false)
set_optimizer_attribute(edmodel, "OutputFlag", 0)
# # modify objective function
# @objective(edmodel, Min, 0.0)
# # modify or add constraints
# @constraint(edmodel, 0.0 <= edmodel[:P][1,1] <= 0.0)

# Solve
timesolve = @elapsed begin
    UCcost, EDcost = STESTS.solving(
        params,
        NDay,
        strategic,
        DADBids,
        DACBids,
        RTDBids,
        RTCBids,
        ucmodel,
        ucpmodel,
        edmodel,
        output_folder,
        PriceCap,
        bidmodels = storagebidmodels,
        ESSeg = ESSeg,
        ESMC = ESMC,
        UCHorizon = UCHorizon,
        EDHorizon = EDHorizon,
        EDSteps = EDSteps,
        BAWindow = BAWindow,
        VOLL = VOLL,
        RM = RM,
        FuelAdjustment = FuelAdjustment,
        ErrorAdjustment = ErrorAdjustment,
        LoadAdjustment = LoadAdjustment,
    )
end
@info "Solving took $timesolve seconds."

println("The UC cost is: ", sum(UCcost))
println("The ED cost is: ", sum(EDcost))
