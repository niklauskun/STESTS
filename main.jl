using STESTS, JuMP, Gurobi, CSV, DataFrames, Statistics, SMTPClient

Year = 2022
Cap = 5
# Read data from .jld2 file 
params = STESTS.read_jld2(
    "./data/ADS2032_" * "$Cap" * "GWBES_BS_AggES_" * "$Year" * "_fixed.jld2",
)
StrategicES = true
ControlES = true
LDESRatio = [0.0, 0.0, 0.0, 0.0] # Ratios of long duration storage capacity to current BESS capacity
LDESDur = [4, 10, 24, 100]
LDESEta = [0.90, 0.75, 0.75, 0.75] # Do not set at 0.80 which is PHS efficiency
LDESMC = [20.0, 10.0, 10.0, 10.0] # Long duration storage marginal cost, $/MWh
FORB = true
seed = 123
heto = false
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
BAWindow = Int(0) # bid-ahead window (number of 5-min intervals, 12-1hr, 48-4hr)
# Define the quadratic function
function quadratic_function(x)
    a = 1 / 9000
    b = 29 / 30
    c = 300
    return min(a * x^2 + b * x + c, 2000)
end

# function quadratic_function(x)
#     a = 5.0
#     b = 300.0
#     return min(a * x + b, 9000.0)
# end

# Generate the array of x values
x_values = 0:100:4900

# Compute the quadratic function for each x value and store the results in an array
y_values = [quadratic_function(x) for x in x_values]
PriceCap = repeat(
    repeat(y_values', outer = (size(params.UCL, 2), 1)),
    outer = (1, 1, EDHorizon),
)
# PriceCap = repeat(
#     repeat(
#         (range(220, stop = 1000, length = 40))',
#         outer = (size(params.UCL, 2), 1),
#     ),
#     outer = (1, 1, EDHorizon),
# )
# PriceCap = repeat(
#     repeat(fill(2000.0, 50)', outer = (size(params.UCL, 2), 1)),
#     outer = (1, 1, EDHorizon),
# )
FuelAdjustment = 2.0
NLCAdjustment = 1.2
ErrorAdjustment = 0.25
LoadAdjustment = 1.0
SegmentAdjustment = [1.0, 2.0, 2.0, 2.0, 2.0]
params.GSMC = params.GSMC .* SegmentAdjustment'
ESPeakBidAdjustment = 1.0
ESPeakBid = 100.0
BSESCbidAdjustment = 0.5
GSMCSeg = join(SegmentAdjustment, "-")
if Year == 2022
    ESAdjustment = 1.0
elseif Year == 2030
    ESAdjustment = 2.7
elseif Year == 2040
    ESAdjustment = 1.28
elseif Year == 2050
    ESAdjustment = 1.3
end

LDESRatio_str = join(LDESRatio, "-")
LDESDur_str = join(LDESDur, "-")
LDESEta_str = join(LDESEta, "-")

output_folder =
    "output/Strategic/TestNewCost/" *
    "$Cap" *
    "GW_ED" *
    "$EDHorizon" *
    "_Strategic_" *
    "$StrategicES" *
    "_FORB_" *
    "$FORB" *
    "_ratio" *
    "$ratio" *
    "_Seg" *
    "$ESSeg"
# *
# "_BSESCbid" *
# "$BSESCbidAdjustment" *
# "_LDESRatio_" *
# "$LDESRatio_str" *
# "_LDESDur_" *
# "$LDESDur_str" *
# "_LDESEta_" *
# "$LDESEta_str"

mkpath(output_folder)

model_base_folder =
    "models/" * "$Cap" * "GW/BAW" * "$BAWindow" * "EDH" * "$EDHorizon"

# Update strategic storage scale base on set ratio
storagebidmodels = []
if StrategicES
    mkpath(output_folder * "/Strategic")
    mkpath(output_folder * "/NStrategic")

    STESTS.add_long_duration_storage!(
        params,
        LDESRatio,
        LDESDur,
        LDESEta,
        LDESMC,
        output_folder,
    )

    STESTS.update_battery_storage!(
        params,
        ControlES,
        ratio,
        output_folder,
        heto,
        ESAdjustment,
        LDESEta,
    )

    # STESTS.update_long_duration_storage!(
    #     params,
    #     LDESRatio,
    #     LDESDur,
    #     output_folder,
    # )

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
    StrategicES = StrategicES,
    Horizon = UCHorizon, # optimization horizon for unit commitment model, 24 hours for WECC data, 4 hours for 3-bus test data
    VOLL = VOLL, # value of lost load, $/MWh
    RM = RM, # reserve margin
    FuelAdjustment = FuelAdjustment,
    NLCAdjustment = NLCAdjustment,
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
    StrategicES = StrategicES,
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
        StrategicES,
        FORB,
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
        LDESEta = LDESEta,
        ESSeg = ESSeg,
        UCHorizon = UCHorizon,
        EDHorizon = EDHorizon,
        EDSteps = EDSteps,
        BAWindow = BAWindow,
        VOLL = VOLL,
        RM = RM,
        FuelAdjustment = FuelAdjustment,
        NLCAdjustment = NLCAdjustment,
        ErrorAdjustment = ErrorAdjustment,
        LoadAdjustment = LoadAdjustment,
        ESPeakBidAdjustment = ESPeakBidAdjustment,
        ESPeakBid = ESPeakBid,
        BSESCbidAdjustment = BSESCbidAdjustment,
        seed = seed,
    )
end
@info "Solving took $timesolve seconds."

println("The UC cost is: ", sum(UCcost))
println("The ED cost is: ", sum(EDcost))
