using JuMP, Gurobi

function economicdispatch(
    params::STESTS.ModelParams,
    PriceCap::Array{Float64}; # value of lost load;
    ESSeg::Int = 1,
    Horizon::Int = 1, # planning horizon
    Steps::Int = 12, # planning steps
    FuelAdjustment::Float64 = 1.0, # fuel adjustment
)::JuMP.Model
    GSMC = repeat(params.GSMC, outer = (1, 1, Horizon)) # segment marginal cost of generators, repeat by EDHorizon
    EDL = convert(Matrix{Float64}, params.EDL[1:Horizon, :]')
    HAvail = convert(Matrix{Float64}, params.EDHAvail[1:Horizon, :]')
    # RAvail = convert(Matrix{Float64}, params.EDRAvail[1:Horizon, :]')
    SAvail = convert(Matrix{Float64}, params.EDSAvail[1:Horizon, :]')
    WAvail = convert(Matrix{Float64}, params.EDWAvail[1:Horizon, :]')
    U = convert(Array{Int64,1}, params.GPIni .!= 0)

    ntimepoints = Horizon # number of time points
    nbus = size(EDL, 1) # number of buses
    ntrans = size(params.transmap, 1) # number of transmission lines
    nucgen = size(params.genmap, 1) # number of conventional generators
    ngucs = size(GSMC, 2) # number of generator segments
    # nhydro = size(params.hydromap, 1) # number of hydro generators
    # nrenewable = size(params.renewablemap, 1) # number of renewable generators
    nstorage = size(params.storagemap, 1) # number of storage units

    ESegSOCini = Dict()

    # Fill ESegSOCini
    for i in 1:nstorage
        excess_soc = params.ESOCini[i]
        segment_increment = params.ESOC[i] / ESSeg  # Maximum increment per segment

        for s in 1:ESSeg
            if excess_soc > 0
                increment = min(segment_increment, excess_soc)
                ESegSOCini[i, s] = increment
                excess_soc -= increment
            else
                ESegSOCini[i, s] = 0
            end
        end
    end

    # Define model
    edmodel = Model()

    # Define decision variables
    @variable(edmodel, f[1:ntrans, 1:ntimepoints]) # Transmission flow
    @variable(edmodel, θ[1:nbus, 1:ntimepoints]) # Phase angle 
    @variable(edmodel, guc[1:nucgen, 1:ntimepoints] >= 0) # Generator output
    # @variable(edmodel, gh[1:nhydro, 1:ntimepoints] >= 0) # Hydro output
    @variable(edmodel, gh[1:nbus, 1:ntimepoints] >= 0) # Hydro output
    # @variable(edmodel, gr[1:nrenewable, 1:ntimepoints] >= 0) # Renewable output
    @variable(edmodel, gs[1:nbus, 1:ntimepoints] >= 0) # Solar output
    @variable(edmodel, gw[1:nbus, 1:ntimepoints] >= 0) # Wind output
    @variable(edmodel, 0 <= s[1:nbus, 1:40, 1:ntimepoints] <= 100) # Slack variable
    # @variable(edmodel, c[1:nstorage, 1:ntimepoints] >= 0) # Storage charging
    # @variable(edmodel, d[1:nstorage, 1:ntimepoints] >= 0) # Storage discharging
    # @variable(edmodel, e[1:nstorage, 1:ntimepoints] >= 0) # Storage energy level
    @variable(edmodel, c[1:nstorage, 1:ESSeg, 1:ntimepoints] >= 0) # Storage charging
    @variable(edmodel, d[1:nstorage, 1:ESSeg, 1:ntimepoints] >= 0) # Storage discharging
    @variable(edmodel, e[1:nstorage, 1:ESSeg, 1:ntimepoints] >= 0) # Storage energy level
    @expression(
        edmodel,
        totalc[i = 1:nstorage, t = 1:ntimepoints],
        sum(c[i, :, t])
    )
    @expression(
        edmodel,
        totald[i = 1:nstorage, t = 1:ntimepoints],
        sum(d[i, :, t])
    )
    @expression(
        edmodel,
        totale[i = 1:nstorage, t = 1:ntimepoints],
        sum(e[i, :, t])
    )
    @expression(edmodel, totals[z = 1:nbus, t = 1:ntimepoints], sum(s[z, :, t]))
    @variable(edmodel, gucs[1:nucgen, 1:ngucs, 1:ntimepoints] >= 0) # Conventional generator segment output

    # Define objective function and constraints， no-load and start-up cost will be added later
    @objective(
        edmodel,
        Min,
        sum(FuelAdjustment * params.GMC .* guc) / Steps +
        sum(FuelAdjustment * GSMC .* gucs) / Steps +
        sum(200 .* d - 0 .* c) / Steps +
        sum(PriceCap .* s) / Steps -
        sun(50 .* gh + 50 .* gs + 50 .* gw) / Steps
    )

    # Bus wise load balance constraints with transmission
    # @constraint(
    #     edmodel,
    #     LoadBalance[z = 1:nbus, t = 1:ntimepoints],
    #     sum(params.genmap[:, z] .* guc[:, t]) +
    #     sum(params.hydromap[:, z] .* gh[:, t]) +
    #     sum(params.renewablemap[:, z] .* gr[:, t]) +
    #     sum(params.storagemap[:, z] .* totald[:, t]) -
    #     sum(params.storagemap[:, z] .* totalc[:, t]) +
    #     sum(params.transmap[:, z] .* f[:, t]) +
    #     s[z, t] == EDL[z, t]
    # )
    @constraint(
        edmodel,
        LoadBalance[z = 1:nbus, t = 1:ntimepoints],
        sum(params.genmap[:, z] .* guc[:, t]) +
        gh[z, t] +
        gs[z, t] +
        gw[z, t] +
        sum(params.storagemap[:, z] .* totald[:, t]) -
        sum(params.storagemap[:, z] .* totalc[:, t]) +
        sum(params.transmap[:, z] .* f[:, t]) +
        sum(s[z, :, t]) == EDL[z, t]
    )

    # Load balance constraints without transmission
    # @constraint(
    #     edmodel,
    #     LoadBalance[t = 1:ntimepoints],
    #     sum(guc[:, t]) + sum(gh[:, t]) + sum(gr[:, t]) + sum(s[:, t]) ==
    #     sum(EDL[:, t])
    # )

    # System reserve constraints
    # @constraint(
    #     edmodel,
    #     Reserve[t = 1:ntimepoints],
    #     sum(guc[:, t]) +
    #     sum(gh[:, t]) +
    #     sum(gr[:, t]) >= (1 + RM) * sum(EDL[:, t])
    # )

    # # Transmission capacity limits
    @constraint(
        edmodel,
        TXCapTo[l = 1:ntrans, t = 1:ntimepoints],
        f[l, t] <= params.TFmax[l]
    )
    @constraint(
        edmodel,
        TXCapFrom[l = 1:ntrans, t = 1:ntimepoints],
        f[l, t] >= -params.TFmax[l]
    )

    # # DCOPF constraints
    # set reference angle
    @constraint(edmodel, REFBUS[t = 1:ntimepoints], θ[1, t] == 0)
    @constraint(
        edmodel,
        DCOPTX[l = 1:ntrans, t = 1:ntimepoints],
        f[l, t] ==
        params.TX[l] * (
            θ[(findfirst(x -> x == 1, params.transmap[l, :])), t] -
            θ[(findfirst(x -> x == -1, params.transmap[l, :])), t]
        )
    )

    # Power flow constraints
    # @constraint(
    #     ucmodel,
    #     PowerFlow[i = 1:ntrans, t = 1:ntimep oints],
    #     f[i, t] ==
    #     (transmap[i, 1] - transmap[i, 2]) *
    #     TX[i] *
    #     (guc[transmap[i, 1], t] - guc[transmap[i, 2], t])
    # )

    # Storage charge and discharge constraints
    # @constraint(
    #     edmodel,
    #     StorageCharge[i = 1:nstorage, t = 1:ntimepoints],
    #     c[i, t] <= params.EPC[i]
    # )

    # @constraint(
    #     edmodel,
    #     StorageDischarge[i = 1:nstorage, t = 1:ntimepoints],
    #     d[i, t] <= params.EPD[i]
    # )

    @constraint(
        edmodel,
        StorageCharge[i = 1:nstorage, t = 1:ntimepoints],
        totalc[i, t] <= params.EPC[i]
    )

    @constraint(
        edmodel,
        StorageDischarge[i = 1:nstorage, t = 1:ntimepoints],
        totald[i, t] <= params.EPD[i]
    )

    # Storage energy level constraints
    # @constraint(
    #     edmodel,
    #     StorageSOCCap[i = 1:nstorage, t = 1:ntimepoints],
    #     e[i, t] <= params.ESOC[i]
    # )

    @constraint(
        edmodel,
        StorageSOCCap[i = 1:nstorage, t = 1:ntimepoints],
        totale[i, t] <= params.ESOC[i]
    )

    @constraint(
        edmodel,
        SegStorageSOCCap[i = 1:nstorage, s = 1:ESSeg, t = 1:ntimepoints],
        e[i, s, t] <= params.ESOC[i] / ESSeg
    )

    @constraint(
        edmodel,
        StorageSOCIni[i = 1:nstorage, s = 1:ESSeg],
        e[i, s, 1] ==
        ESegSOCini[i, s] + c[i, s, 1] * params.Eeta[i] / Steps -
        d[i, s, 1] / params.Eeta[i] / Steps
    )

    @constraint(
        edmodel,
        StorageSOC[i = 1:nstorage, s = 1:ESSeg, t = 2:ntimepoints],
        e[i, s, t] ==
        e[i, s, t-1] + c[i, s, 1] * params.Eeta[i] / Steps -
        d[i, s, 1] / params.Eeta[i] / Steps
    )

    # Storage SOC evolution constraints
    # @constraint(
    #     edmodel,
    #     StorageSOCIni[i = 1:nstorage],
    #     e[i, 1] ==
    #     params.ESOCini[i] + c[i, 1] * params.Eeta[i] / Steps -
    #     d[i, 1] / params.Eeta[i] / Steps
    # )

    # @constraint(
    #     edmodel,
    #     StorageSOCIni[i = 1:nstorage],
    #     totale[i, 1] ==
    #     params.ESOCini[i] + totalc[i, 1] * params.Eeta[i] / Steps -
    #     totald[i, 1] / params.Eeta[i] / Steps
    # )

    # @constraint(
    #     edmodel,
    #     StorageSOC[i = 1:nstorage, t = 2:ntimepoints],
    #     e[i, t] ==
    #     e[i, t-1] + c[i, t] * params.Eeta[i] / Steps -
    #     d[i, t] / params.Eeta[i] / Steps
    # )

    # @constraint(
    #     edmodel,
    #     StorageSOC[i = 1:nstorage, t = 2:ntimepoints],
    #     totale[i, t] ==
    #     totale[i, t-1] + totalc[i, t] * params.Eeta[i] / Steps -
    #     totald[i, t] / params.Eeta[i] / Steps
    # )

    # Conventional generator segment constraints
    @constraint(
        edmodel,
        UCGenSeg1[i = 1:nucgen, t = 1:ntimepoints],
        guc[i, t] == U[i] * params.GPmin[i] + sum(gucs[i, :, t])
    )

    @constraint(
        edmodel,
        UCGenSeg2[i = 1:nucgen, j = 1:ngucs, t = 1:ntimepoints],
        gucs[i, j, t] <= params.GINCPmax[i, j]
    )

    # Conventional generator capacity limits
    @constraint(
        edmodel,
        UCCapU[i = 1:nucgen, t = 1:Horizon],
        guc[i, t] <= U[i] * params.GPmax[i]
    )
    @constraint(
        edmodel,
        UCCapL[i = 1:nucgen, t = 1:Horizon],
        guc[i, t] >= U[i] * params.GPmin[i]
    )
    # Hydro and renewable capacity limits
    # @constraint(
    #     edmodel,
    #     HCap[i = 1:nhydro, t = 1:Horizon],
    #     gh[i, t] <= HAvail[i, t]
    # )
    @constraint(
        edmodel,
        HCap[z = 1:nbus, t = 1:Horizon],
        gh[z, t] <= HAvail[z, t]
    )
    # @constraint(
    #     edmodel,
    #     HCap2[z = 1:nbus, t = 1:Horizon],
    #     gh[z, t] >= params.HPmin[z]
    # )
    # @constraint(
    #     edmodel,
    #     ReCap[i = 1:nrenewable, t = 1:Horizon],
    #     gr[i, t] <= RAvail[i, t]
    # )
    @constraint(
        edmodel,
        SCap[z = 1:nbus, t = 1:Horizon],
        gs[z, t] <= SAvail[z, t]
    )
    @constraint(
        edmodel,
        WCap[z = 1:nbus, t = 1:Horizon],
        gw[z, t] <= WAvail[z, t]
    )
    # Ramping limits, set as one hour ramping limits for initial generation output
    @constraint(
        edmodel,
        RUIni[i = 1:nucgen],
        guc[i, 1] - params.GPIni[i] <= params.GRU[i]
    )
    @constraint(
        edmodel,
        RDIni[i = 1:nucgen],
        params.GPIni[i] - guc[i, 1] <= params.GRD[i]
    )
    if Horizon > 1
        @constraint(
            edmodel,
            RU[i = 1:nucgen, t = 2:Horizon],
            guc[i, t] - guc[i, t-1] <= params.GRU[i] / Steps
        )

        @constraint(
            edmodel,
            RD[i = 1:nucgen, t = 2:Horizon],
            guc[i, t-1] - guc[i, t] <= params.GRD[i] / Steps
        )
    end
    return edmodel
end