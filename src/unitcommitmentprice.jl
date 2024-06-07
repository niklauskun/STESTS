using JuMP
"""
    unitcommitmentprice()::JuMP.Model

Read system parameters and construct unit commitment model.
Nik's note: In julia vectorized functions are not requireed for performance.
            For readability, I write constriants in for loops.

# Methods
unitcommitment(L::Matrix{Float64})::JuMP.Model -> Multi-bus ucmodel
unitcommitment(L::Vector{Float64})::JuMP.Model -> Single-bus ucmodel
"""
function unitcommitmentprice(
    params::STESTS.ModelParams;
    Horizon::Int = 24, # planning horizon
    VOLL::Float64 = 1000.0, # value of lost load
    RM::Float64 = 0.03, # reserve margin
    FuelAdjustment::Float64 = 1.0, # fuel adjustment
)::JuMP.Model
    GSMC = repeat(params.GSMC, outer = (1, 1, Horizon))
    UCL = convert(Matrix{Float64}, params.UCL[1:Horizon, :]')
    HAvail = convert(Matrix{Float64}, params.HAvail[1:Horizon, :]')
    # RAvail = convert(Matrix{Float64}, params.RAvail[1:Horizon, :]')
    SAvail = convert(Matrix{Float64}, params.SAvail[1:Horizon, :]')
    WAvail = convert(Matrix{Float64}, params.WAvail[1:Horizon, :]')

    ntimepoints = Horizon # number of time points
    nbus = size(UCL, 1) # number of buses
    ntrans = size(params.transmap, 1) # number of transmission lines
    nucgen = size(params.genmap, 1) # number of conventional generators
    ngucs = size(GSMC, 2) # number of generator segments
    # nhydro = size(params.hydromap, 1) # number of hydro generators
    # nrenewable = size(params.renewablemap, 1) # number of renewable generators
    nstorage = size(params.storagemap, 1) # number of storage units
    U = zeros(Int, size(params.GPIni, 1), Horizon)
    V = zeros(Int, size(params.GPIni, 1), Horizon)
    W = zeros(Int, size(params.GPIni, 1), Horizon)

    # Define model
    ucpmodel = Model()

    # Define decision variables
    @variable(ucpmodel, f[1:ntrans, 1:ntimepoints]) # Transmission flow
    @variable(ucpmodel, θ[1:nbus, 1:ntimepoints]) # Phase angle 
    @variable(ucpmodel, guc[1:nucgen, 1:ntimepoints] >= 0) # Generator output
    # @variable(ucpmodel, gh[1:nhydro, 1:ntimepoints] >= 0) # Hydro output
    @variable(ucpmodel, gh[1:nbus, 1:ntimepoints] >= 0) # Hydro output
    # @variable(ucpmodel, gr[1:nrenewable, 1:ntimepoints] >= 0) # Renewable output
    @variable(ucpmodel, gs[1:nbus, 1:ntimepoints] >= 0) # Solar output
    @variable(ucpmodel, gw[1:nbus, 1:ntimepoints] >= 0) # Wind output
    @variable(ucpmodel, s[1:nbus, 1:ntimepoints] >= 0) # Slack variable
    @variable(ucpmodel, c[1:nstorage, 1:ntimepoints] >= 0) # Storage charging
    @variable(ucpmodel, d[1:nstorage, 1:ntimepoints] >= 0) # Storage discharging
    @variable(ucpmodel, e[1:nstorage, 1:ntimepoints] >= 0) # Storage energy level
    @variable(ucpmodel, grr[1:nucgen, 1:ntimepoints] >= 0) # Conventional generator reserve
    @variable(ucpmodel, gucs[1:nucgen, 1:ngucs, 1:ntimepoints] >= 0) # Conventional generator segment output

    # Define objective function and constraints
    @objective(
        ucpmodel,
        Min,
        sum(FuelAdjustment * params.GMC .* guc) +
        sum(FuelAdjustment * GSMC .* gucs) +
        sum(300.0 .* d - 0.0 .* c) +
        sum(VOLL .* s) - sum(50 .* gh + 50 .* gs + 50 .* gw)
    )

    # Bus wise load balance constraints with transmission
    # @constraint(
    #     ucpmodel,
    #     LoadBalance[z = 1:nbus, h = 1:ntimepoints],
    #     sum(params.genmap[:, z] .* guc[:, h]) +
    #     sum(params.hydromap[:, z] .* gh[:, h]) +
    #     sum(params.renewablemap[:, z] .* gr[:, h]) +
    #     sum(params.storagemap[:, z] .* d[:, h]) -
    #     sum(params.storagemap[:, z] .* c[:, h]) +
    #     sum(params.transmap[:, z] .* f[:, h]) +
    #     s[z, h] == UCL[z, h]
    # )
    @constraint(
        ucpmodel,
        LoadBalance[z = 1:nbus, h = 1:ntimepoints],
        sum(params.genmap[:, z] .* guc[:, h]) +
        gh[z, h] +
        gs[z, h] +
        gw[z, h] +
        sum(params.storagemap[:, z] .* d[:, h]) -
        sum(params.storagemap[:, z] .* c[:, h]) +
        sum(params.transmap[:, z] .* f[:, h]) +
        s[z, h] == UCL[z, h]
    )

    # Load balance constraints without transmission
    # @constraint(
    #     ucmodel,
    #     LoadBalance[h = 1:ntimepoints],
    #     sum(guc[:, h]) + sum(gh[:, h]) + sum(gr[:, h]) + sum(s[:, h]) ==
    #     sum(UCL[:, h])
    # )

    # System reserve constraints
    @constraint(
        ucpmodel,
        UnitReserve1[i = 1:nucgen, h = 1:ntimepoints],
        grr[i, h] <= params.GPmax[i] * U[i, h] - guc[i, h]
    )

    @constraint(
        ucpmodel,
        UnitReserve2[i = 1:nucgen, h = 1:ntimepoints],
        grr[i, h] <= params.GRU[i] / 6
    )

    @constraint(
        ucpmodel,
        Reserve[h = 1:ntimepoints],
        sum(grr[:, h]) >=
        RM * sum(UCL, dims = 1)[h] +
        0.05 * sum(gw[:, h]) +
        0.05 * sum(gs[:, h])
    )

    # # Transmission capacity limits
    @constraint(
        ucpmodel,
        TXCapTo[l = 1:ntrans, h = 1:ntimepoints],
        f[l, h] <= params.TFmax[l]
    )
    @constraint(
        ucpmodel,
        TXCapFrom[l = 1:ntrans, h = 1:ntimepoints],
        f[l, h] >= -params.TFmax[l]
    )

    # # DCOPF constraints
    # set reference angle
    @constraint(ucpmodel, REFBUS[h = 1:ntimepoints], θ[1, h] == 0)
    @constraint(
        ucpmodel,
        DCOPTX[l = 1:ntrans, h = 1:ntimepoints],
        f[l, h] ==
        params.TX[l] * (
            θ[(findfirst(x -> x == 1, params.transmap[l, :])), h] -
            θ[(findfirst(x -> x == -1, params.transmap[l, :])), h]
        )
    )

    # Storage charge and discharge constraints
    @constraint(
        ucpmodel,
        StorageCharge[i = 1:nstorage, h = 1:ntimepoints],
        c[i, h] <= params.EPC[i]
    )

    @constraint(
        ucpmodel,
        StorageDischarge[i = 1:nstorage, h = 1:ntimepoints],
        d[i, h] <= params.EPD[i]
    )

    # Storage energy level constraints
    @constraint(
        ucpmodel,
        StorageSOCCap[i = 1:nstorage, h = 1:ntimepoints],
        e[i, h] <= params.ESOC[i]
    )

    # Storage SOC evolution constraints
    @constraint(
        ucpmodel,
        StorageSOCIni[i = 1:nstorage],
        e[i, 1] ==
        params.ESOCini[i] + c[i, 1] * params.Eeta[i] - d[i, 1] / params.Eeta[i]
    )

    @constraint(
        ucpmodel,
        StorageSOC[i = 1:nstorage, h = 2:ntimepoints],
        e[i, h] ==
        e[i, h-1] + c[i, h] * params.Eeta[i] - d[i, h] / params.Eeta[i]
    )

    @constraint(
        ucpmodel,
        StorageSOCEnd[i = 1:nstorage, h = 1:ntimepoints; h % 24 == 0],  # Apply constraint for every h divisible by 24
        e[i, h] >= params.ESOCini[i]
    )

    # Conventional generator must run constraints
    # @constraint(
    #     ucpmodel,
    #     UCMustRun[i = 1:nucgen, h = 1:ntimepoints],
    #     u[i, h] >= GMustRun[i]
    # )

    # Conventional generator segment constraints
    @constraint(
        ucpmodel,
        UCGenSeg1[i = 1:nucgen, h = 1:ntimepoints],
        guc[i, h] == U[i, h] * params.GPmin[i] + sum(gucs[i, :, h])
    )

    @constraint(
        ucpmodel,
        UCGenSeg2[i = 1:nucgen, j = 1:ngucs, h = 1:ntimepoints],
        gucs[i, j, h] <= params.GINCPmax[i, j]
    )

    # Conventional generator capacity limits
    @constraint(
        ucpmodel,
        UCCapU[i = 1:nucgen, h = 1:Horizon],
        guc[i, h] <= U[i, h] * params.GPmax[i]
    )
    @constraint(
        ucpmodel,
        UCCapL[i = 1:nucgen, h = 1:Horizon],
        guc[i, h] >= U[i, h] * params.GPmin[i]
    )
    # Hydro and renewable capacity limits
    # @constraint(
    #     ucpmodel,
    #     HCap[i = 1:nhydro, h = 1:Horizon],
    #     gh[i, h] <= HAvail[i, h]
    # )
    @constraint(
        ucpmodel,
        HCap[z = 1:nbus, h = 1:Horizon],
        gh[z, h] <= HAvail[z, h]
    )
    # @constraint(
    #     ucpmodel,
    #     HCap2[z = 1:nbus, h = 1:Horizon],
    #     gh[z, h] >= params.HPmin[z]
    # )
    # @constraint(
    #     ucpmodel,
    #     ReCap[i = 1:nrenewable, h = 1:Horizon],
    #     gr[i, h] <= RAvail[i, h]
    # )
    @constraint(
        ucpmodel,
        SCap[z = 1:nbus, h = 1:Horizon],
        gs[z, h] <= SAvail[z, h]
    )
    @constraint(
        ucpmodel,
        WCap[z = 1:nbus, h = 1:Horizon],
        gw[z, h] <= WAvail[z, h]
    )
    # Ramping limits
    # @constraint(ucmodel, RUIni[i = 1:nucgen], guc[i, 1] - GPini[i] <= GRU[i])
    # @constraint(ucmodel, RDIni[i = 1:nucgen], GPini[i] - guc[i, 1] <= GRD[i])
    # @constraint(
    #     ucmodel,
    #     RU[i = 1:nucgen, h = 1:Horizon-1],
    #     guc[i, h+1] - guc[i, h] <= GRU[i]
    # )
    # @constraint(
    #     ucmodel,
    #     RD[i = 1:nucgen, h = 1:Horizon-1],
    #     guc[i, h] - guc[i, h+1] <= GRD[i]
    # )
    @constraint(
        ucpmodel,
        RUIni[i = 1:nucgen],
        guc[i, 1] - params.GPIni[i] <=
        params.GRU[i] + params.GPmin[i] * V[i, 1]
    )
    @constraint(
        ucpmodel,
        RDIni[i = 1:nucgen],
        params.GPIni[i] - guc[i, 1] <=
        params.GRD[i] + params.GPmin[i] * W[i, 1]
    )
    @constraint(
        ucpmodel,
        RU[i = 1:nucgen, h = 1:Horizon-1],
        guc[i, h+1] - guc[i, h] <= params.GRU[i] + params.GPmin[i] * V[i, h+1]
    )
    @constraint(
        ucpmodel,
        RD[i = 1:nucgen, h = 1:Horizon-1],
        guc[i, h] - guc[i, h+1] <= params.GRD[i] + params.GPmin[i] * W[i, h+1]
    )

    # State transition constraints
    # @constraint(
    #     ucpmodel,
    #     ST0[i = 1:nucgen],
    #     U[i, 1] - UInput[i] == V[i, 1] - W[i, 1]
    # )

    # @constraint(
    #     ucpmodel,
    #     ST1[i = 1:nucgen, h = 2:Horizon],
    #     U[i, h] - U[i, h-1] == V[i, h] - W[i, h]
    # )

    # @constraint(
    #     ucmodel,
    #     ST1[i = 1:nucgen, h = 1:Horizon-1],
    #     u[i, h+1] - u[i, h] == v[i, h] - w[i, h]
    # )

    # @constraint(
    #     ucmodel,
    #     ST2[i = 1:nucgen, h = 1:Horizon-1],
    #      v[i, h] + w[i, h] <= 1
    # )

    # Minimum up/down time constraints
    # @constraint(
    #     ucpmodel,
    #     UTime[i = 1:nucgen, h = 1:Horizon],
    #     sum(V[i, max(1, h - GUT[i] + 1):h]) <= U[i, h]
    # )
    # @constraint(
    #     ucpmodel,
    #     DTime[i = 1:nucgen, h = 1:Horizon],
    #     sum(W[i, max(1, h - GDT[i] + 1):h]) <= 1 - U[i, h]
    # )

    # @constraint(
    #     ucpmodel, 
    #     UTimeIni[i = 1:nucgen],
    #     sum(0 .* U[i, :]) == SU[i]
    # )

    # @constraint(
    #     ucpmodel,
    #     DTimeIni[i = 1:nucgen],
    #     sum(0 .* U[i, :]) == SD[i]
    # )

    return ucpmodel
end