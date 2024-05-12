using JuMP
"""
    unitcommitment()::JuMP.Model

Read system parameters and construct unit commitment model.
Nik's note: In julia vectorized functions are not requireed for performance.
            For readability, I write constriants in for loops.

# Methods
unitcommitment(L::Matrix{Float64})::JuMP.Model -> Multi-bus ucmodel
unitcommitment(L::Vector{Float64})::JuMP.Model -> Single-bus ucmodel
"""
function unitcommitment(
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
    UInput = convert(Array{Int64,1}, params.GPIni .!= 0)
    SU = zeros(Int, size(params.GPIni, 1)) # initial generator must on time
    SD = zeros(Int, size(params.GPIni, 1)) # initial generator down on time

    ntimepoints = Horizon # number of time points
    nbus = size(UCL, 1) # number of buses
    ntrans = size(params.transmap, 1) # number of transmission lines
    nucgen = size(params.genmap, 1) # number of conventional generators
    ngucs = size(GSMC, 2) # number of generator segments
    # nhydro = size(params.hydromap, 1) # number of hydro generators
    # nrenewable = size(params.renewablemap, 1) # number of renewable generators
    nstorage = size(params.storagemap, 1) # number of storage units

    # Define model
    ucmodel = Model()

    # Define decision variables
    @variable(ucmodel, f[1:ntrans, 1:ntimepoints]) # Transmission flow
    @variable(ucmodel, θ[1:nbus, 1:ntimepoints]) # Phase angle 
    @variable(ucmodel, guc[1:nucgen, 1:ntimepoints] >= 0) # Generator output
    @variable(ucmodel, u[1:nucgen, 1:ntimepoints], Bin) # Conventional generator status, 1 if on, 0 if off
    @variable(ucmodel, v[1:nucgen, 1:ntimepoints], Bin) # Conventional generator start-up decision, 1 if start-up, 0 otherwise
    @variable(ucmodel, w[1:nucgen, 1:ntimepoints], Bin) # Conventional generator shut-down decision, 1 if shut-down, 0 otherwise
    # @variable(ucmodel, gh[1:nhydro, 1:ntimepoints] >= 0) # Hydro output
    @variable(ucmodel, gh[1:nbus, 1:ntimepoints] >= 0) # Hydro output
    # @variable(ucmodel, gr[1:nrenewable, 1:ntimepoints] >= 0) # Renewable output
    @variable(ucmodel, gs[1:nbus, 1:ntimepoints] >= 0) # Solar output
    @variable(ucmodel, gw[1:nbus, 1:ntimepoints] >= 0) # Wind output
    @variable(ucmodel, s[1:nbus, 1:ntimepoints] >= 0) # Slack variable
    @variable(ucmodel, c[1:nstorage, 1:ntimepoints] >= 0) # Storage charging
    @variable(ucmodel, d[1:nstorage, 1:ntimepoints] >= 0) # Storage discharging
    @variable(ucmodel, e[1:nstorage, 1:ntimepoints] >= 0) # Storage energy level
    @variable(ucmodel, grr[1:nucgen, 1:ntimepoints] >= 0) # Conventional generator reserve
    @variable(ucmodel, gucs[1:nucgen, 1:ngucs, 1:ntimepoints] >= 0) # Conventional generator segment output

    # Define objective function and constraints
    @objective(
        ucmodel,
        Min,
        sum(
            FuelAdjustment * params.GMC .* guc +
            params.GNLC .* u +
            params.GSUC .* v,
        ) +
        sum(FuelAdjustment * GSMC .* gucs) +
        sum(300.0 .* d - 0.0 .* c) +
        sum(VOLL .* s) - sun(50 .* gh + 50 .* gs + 50 .* gw) / Steps
    )

    # Bus wise load balance constraints with transmission
    # @constraint(
    #     ucmodel,
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
        ucmodel,
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
        ucmodel,
        UnitReserve1[i = 1:nucgen, h = 1:ntimepoints],
        grr[i, h] <= params.GPmax[i] * u[i, h] - guc[i, h]
    )

    @constraint(
        ucmodel,
        UnitReserve2[i = 1:nucgen, h = 1:ntimepoints],
        grr[i, h] <= params.GRU[i] / 6
    )

    @constraint(
        ucmodel,
        Reserve[h = 1:ntimepoints],
        sum(grr[:, h]) >= RM * sum(UCL, dims = 1)[h]
    )

    # # Transmission capacity limits
    @constraint(
        ucmodel,
        TXCapTo[l = 1:ntrans, h = 1:ntimepoints],
        f[l, h] <= params.TFmax[l]
    )
    @constraint(
        ucmodel,
        TXCapFrom[l = 1:ntrans, h = 1:ntimepoints],
        f[l, h] >= -params.TFmax[l]
    )

    # # DCOPF constraints
    # set reference angle
    @constraint(ucmodel, REFBUS[h = 1:ntimepoints], θ[1, h] == 0)
    @constraint(
        ucmodel,
        DCOPTX[l = 1:ntrans, h = 1:ntimepoints],
        f[l, h] ==
        params.TX[l] * (
            θ[(findfirst(x -> x == 1, params.transmap[l, :])), h] -
            θ[(findfirst(x -> x == -1, params.transmap[l, :])), h]
        )
    )

    # Storage charge and discharge constraints
    @constraint(
        ucmodel,
        StorageCharge[i = 1:nstorage, h = 1:ntimepoints],
        c[i, h] <= params.EPC[i]
    )

    @constraint(
        ucmodel,
        StorageDischarge[i = 1:nstorage, h = 1:ntimepoints],
        d[i, h] <= params.EPD[i]
    )

    # Storage energy level constraints
    @constraint(
        ucmodel,
        StorageSOCCap[i = 1:nstorage, h = 1:ntimepoints],
        e[i, h] <= params.ESOC[i]
    )

    # Storage SOC evolution constraints
    @constraint(
        ucmodel,
        StorageSOCIni[i = 1:nstorage],
        e[i, 1] ==
        params.ESOCini[i] + c[i, 1] * params.Eeta[i] - d[i, 1] / params.Eeta[i]
    )

    @constraint(
        ucmodel,
        StorageSOC[i = 1:nstorage, h = 2:ntimepoints],
        e[i, h] ==
        e[i, h-1] + c[i, h] * params.Eeta[i] - d[i, h] / params.Eeta[i]
    )

    @constraint(
        ucmodel,
        StorageSOCEnd[i = 1:nstorage, h = 1:ntimepoints; h % 24 == 0],  # Apply constraint for every h divisible by 24
        e[i, h] >= params.ESOCini[i]
    )

    # Conventional generator must run constraints
    @constraint(
        ucmodel,
        UCMustRun[i = 1:nucgen, h = 1:ntimepoints],
        u[i, h] >= params.GMustRun[i]
    )

    # Conventional generator segment constraints
    @constraint(
        ucmodel,
        UCGenSeg1[i = 1:nucgen, h = 1:ntimepoints],
        guc[i, h] == u[i, h] * params.GPmin[i] + sum(gucs[i, :, h])
    )

    @constraint(
        ucmodel,
        UCGenSeg2[i = 1:nucgen, j = 1:ngucs, h = 1:ntimepoints],
        gucs[i, j, h] <= params.GINCPmax[i, j]
    )

    # Conventional generator capacity limits
    @constraint(
        ucmodel,
        UCCapU[i = 1:nucgen, h = 1:Horizon],
        guc[i, h] <= u[i, h] * params.GPmax[i]
    )
    @constraint(
        ucmodel,
        UCCapL[i = 1:nucgen, h = 1:Horizon],
        guc[i, h] >= u[i, h] * params.GPmin[i]
    )
    # Hydro and renewable capacity limits
    # @constraint(
    #     ucmodel,
    #     HCap[i = 1:nhydro, h = 1:Horizon],
    #     gh[i, h] <= HAvail[i, h]
    # )
    @constraint(
        ucmodel,
        HCap[z = 1:nbus, h = 1:Horizon],
        gh[z, h] <= HAvail[z, h]
    )
    # @constraint(
    #     ucmodel,
    #     HCap2[z = 1:nbus, h = 1:Horizon],
    #     gh[z, h] >= params.HPmin[z]
    # )
    # @constraint(
    #     ucmodel,
    #     ReCap[i = 1:nrenewable, h = 1:Horizon],
    #     gr[i, h] <= RAvail[i, h]
    # )
    @constraint(
        ucmodel,
        SCap[z = 1:nbus, h = 1:Horizon],
        gs[z, h] <= SAvail[z, h]
    )
    @constraint(
        ucmodel,
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
        ucmodel,
        RUIni[i = 1:nucgen],
        guc[i, 1] - params.GPIni[i] <=
        params.GRU[i] + params.GPmin[i] * v[i, 1]
    )
    @constraint(
        ucmodel,
        RDIni[i = 1:nucgen],
        params.GPIni[i] - guc[i, 1] <=
        params.GRD[i] + params.GPmin[i] * w[i, 1]
    )
    @constraint(
        ucmodel,
        RU[i = 1:nucgen, h = 1:Horizon-1],
        guc[i, h+1] - guc[i, h] <= params.GRU[i] + params.GPmin[i] * v[i, h+1]
    )
    @constraint(
        ucmodel,
        RD[i = 1:nucgen, h = 1:Horizon-1],
        guc[i, h] - guc[i, h+1] <= params.GRD[i] + params.GPmin[i] * w[i, h+1]
    )

    # State transition constraints
    @constraint(
        ucmodel,
        ST0[i = 1:nucgen],
        u[i, 1] - UInput[i] == v[i, 1] - w[i, 1]
    )

    @constraint(
        ucmodel,
        ST1[i = 1:nucgen, h = 2:Horizon],
        u[i, h] - u[i, h-1] == v[i, h] - w[i, h]
    )

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
    @constraint(
        ucmodel,
        UTime[i = 1:nucgen, h = 1:Horizon],
        sum(v[i, max(1, h - params.GUT[i] + 1):h]) <= u[i, h]
    )
    @constraint(
        ucmodel,
        DTime[i = 1:nucgen, h = 1:Horizon],
        sum(w[i, max(1, h - params.GDT[i] + 1):h]) <= 1 - u[i, h]
    )

    @constraint(ucmodel, UTimeIni[i = 1:nucgen], sum(0 .* u[i, :]) == SU[i])

    @constraint(ucmodel, DTimeIni[i = 1:nucgen], sum(0 .* u[i, :]) == SD[i])

    return ucmodel
end