using JuMP, DataFrames, CSV, Gurobi, Statistics

function getLastOneIndex(
    matrix::Array{Int,2},
    vector::Array{Int,1},
    vector2::Array{Int,1},
)
    m, n = size(matrix)
    result = Vector{Int}(undef, m)

    for i in 1:m
        idx = findlast(==(1), matrix[i, :])
        result[i] =
            idx === nothing ? max(0, vector[m] - 24) :
            max(0, vector2[m] - (24 + 1 - idx))
    end

    return result
end

function enforce_strictly_decreasing_vector(v, delta = 1e-5)
    indices = eachindex(v)
    for i in first(indices)+1:last(indices)
        if v[i] >= v[i-1]
            v[i-1] = v[i] + delta
        end
    end
end

function solvingError(model::JuMP.Model, type::String, d = 0, h = 0, t = 0)
    println("Model is infeasible. Starting IIS analysis...")
    compute_conflict!(model)
    iis_model, _ = copy_conflict(model)
    output_file = "diagnose_output_" * type * ".txt"
    open(output_file, "w") do f
        return write(f, string(iis_model))
    end
    print(iis_model)
    if type == "ED"
        error("No optimal solution found for ED at hour $h step $t.")
    elseif type == "UC"
        error("No optimal solution found for UC on day $d.")
    end
end

function setUCConstraints(
    params::STESTS.ModelParams,
    ucmodel::JuMP.Model,
    ucpmodel::JuMP.Model,
    LInput::Matrix{Float64},
    HAvailInput::Matrix{Float64},
    WAvailInput::Matrix{Float64},
    SAvailInput::Matrix{Float64},
    DADBids::Matrix{Float64},
    DAdb::Matrix{Float64},
    DAcb::Matrix{Float64},
    FORB::Bool,
    DAOInput::Matrix{Int},
    UCHorizon::Int = 24,
    RM::Float64 = 0.03,
)
    for h in 1:UCHorizon
        for z in axes(LInput, 1)
            set_normalized_rhs(ucmodel[:LoadBalance][z, h], LInput[z, h])
            set_normalized_rhs(
                ucmodel[:Reserve][h],
                RM * sum(LInput, dims = 1)[h],
            )
            set_normalized_rhs(ucpmodel[:LoadBalance][z, h], LInput[z, h])
            set_normalized_rhs(
                ucpmodel[:Reserve][h],
                RM * sum(LInput, dims = 1)[h],
            )
        end
        for i in axes(HAvailInput, 1)
            set_normalized_rhs(ucmodel[:HCap][i, h], HAvailInput[i, h])
            set_normalized_rhs(ucpmodel[:HCap][i, h], HAvailInput[i, h])
        end
        # for i in axes(RAvailInput, 1)
        #     set_normalized_rhs(ucmodel[:ReCap][i, h], RAvailInput[i, h])
        #     set_normalized_rhs(ucpmodel[:ReCap][i, h], RAvailInput[i, h])
        # end
        for z in axes(SAvailInput, 1)
            set_normalized_rhs(ucmodel[:SCap][z, h], SAvailInput[z, h])
            set_normalized_rhs(ucpmodel[:SCap][z, h], SAvailInput[z, h])
        end
        for z in axes(WAvailInput, 1)
            set_normalized_rhs(ucmodel[:WCap][z, h], WAvailInput[z, h])
            set_normalized_rhs(ucpmodel[:WCap][z, h], WAvailInput[z, h])
        end
        for i in axes(DADBids, 1)
            set_objective_coefficient(ucmodel, ucmodel[:d][i, h], DAdb[i, h])
            set_objective_coefficient(ucmodel, ucmodel[:c][i, h], DAcb[i, h])
            set_objective_coefficient(ucpmodel, ucpmodel[:d][i, h], DAdb[i, h])
            set_objective_coefficient(ucpmodel, ucpmodel[:c][i, h], DAcb[i, h])
        end
        if FORB
            for i in axes(params.GPIni, 1)
                for h in 1:UCHorizon
                    set_normalized_rhs(
                        ucmodel[:UCMustRun][i, h],
                        params.GMustRun[i] * DAOInput[i, h],
                    )
                    set_normalized_rhs(
                        ucmodel[:UCForcedOutage][i, h],
                        DAOInput[i, h],
                    )
                    # set_normalized_rhs(
                    #     ucmodel[:UTime][i, h],
                    #     1 - DAOInput[i, h],
                    # )
                end
            end
        end
    end
end

function writeUCtoCSV(
    params,
    output_folder::String,
    ucmodel,
    Udf,
    Vdf,
    Wdf,
    Sdf,
)
    CSV.write(joinpath(output_folder, "UCCommit.csv"), Udf, append = true)
    CSV.write(joinpath(output_folder, "UCStart.csv"), Vdf, append = true)
    CSV.write(joinpath(output_folder, "UCShut.csv"), Wdf, append = true)
    CSV.write(joinpath(output_folder, "UCSlack.csv"), Sdf, append = true)
    UCGdf = DataFrame(value.(ucmodel[:guc])', :auto)
    CSV.write(joinpath(output_folder, "UCGen.csv"), UCGdf, append = true)
    # UCRedf = DataFrame(value.(ucmodel[:gr])', :auto)
    # CSV.write(
    #     joinpath(output_folder, "UCRenewable.csv"),
    #     UCRedf,
    #     append = true,
    # )
    UCSolardf = DataFrame(value.(ucmodel[:gs])', :auto)
    CSV.write(joinpath(output_folder, "UCSolar.csv"), UCSolardf, append = true)
    UCWinddf = DataFrame(value.(ucmodel[:gw])', :auto)
    CSV.write(joinpath(output_folder, "UCWind.csv"), UCWinddf, append = true)
    UCHdf = DataFrame(value.(ucmodel[:gh])', :auto)
    CSV.write(joinpath(output_folder, "UCHydro.csv"), UCHdf, append = true)
    UCGZone = DataFrame(value.(ucmodel[:guc])' * params.genmap, :auto)
    CSV.write(joinpath(output_folder, "UCGenZone.csv"), UCGZone, append = true)
    # UCHydroZone =
    #     DataFrame(value.(ucmodel[:gh])' * params.hydromap, :auto)
    # CSV.write(
    #     joinpath(output_folder, "UCHydroZone.csv"),
    #     UCHydroZone,
    #     append = true,
    # )
    # UCRenewableZone =
    #     DataFrame(value.(ucmodel[:gr])' * params.renewablemap, :auto)
    # CSV.write(
    #     joinpath(output_folder, "UCRenewableZone.csv"),
    #     UCRenewableZone,
    #     append = true,
    # )
    UCSoCZone = DataFrame(value.(ucmodel[:e])' * params.storagemap, :auto)
    CSV.write(
        joinpath(output_folder, "UCSoCZone.csv"),
        UCSoCZone,
        append = true,
    )
    UCFdf = DataFrame(value.(ucmodel[:f])', :auto)
    return CSV.write(
        joinpath(output_folder, "UCTrans.csv"),
        UCFdf,
        append = true,
    )
end

function calculateDAP(params, ucpmodel, UCHorizon, U, V, W, GPIniInput)
    # Calculate day-ahead price
    for i in axes(params.GPIni, 1)
        for h in 1:UCHorizon
            set_normalized_rhs(
                ucpmodel[:UnitReserve1][i, h],
                params.GPmax[i] * U[i, h],
            )
            set_normalized_rhs(
                ucpmodel[:UCGenSeg1][i, h],
                params.GPmin[i] * U[i, h],
            )
            set_normalized_rhs(
                ucpmodel[:UCCapU][i, h],
                params.GPmax[i] * U[i, h],
            )
            set_normalized_rhs(
                ucpmodel[:UCCapL][i, h],
                params.GPmin[i] * U[i, h],
            )
        end
        for h in 1:(UCHorizon-1)
            set_normalized_rhs(
                ucpmodel[:RU][i, h],
                params.GRU[i] + params.GPmin[i] * V[i, h+1],
            )
            set_normalized_rhs(
                ucpmodel[:RD][i, h],
                params.GRD[i] + params.GPmin[i] * W[i, h+1],
            )
        end
        set_normalized_rhs(
            ucpmodel[:RUIni][i],
            params.GRU[i] + params.GPmin[i] * V[i, 1] + GPIniInput[i],
        )
        set_normalized_rhs(
            ucpmodel[:RDIni][i],
            params.GRD[i] + params.GPmin[i] * W[i, 1] - GPIniInput[i],
        )
    end
end

function setEDConstraints(
    params,
    output_folder,
    edmodel,
    EDV,
    EDW,
    EDU,
    U,
    EDGPIni,
    EDSteps,
    BAWindow,
    strategic,
    FORB,
    bidmodels,
    db,
    cb,
    vrt,
    segment_length,
    ESSeg,
    RTDBids,
    RTCBids,
    ESPeakBid,
    ESPeakBidAdjustment,
    AdjustedEDL,
    AdjustedEDSolar,
    AdjustedEDWind,
    RTOInput,
    all_UCprices_df,
    all_EDprices_df,
    EDHorizon,
    d,
    h,
    t,
    ts,
    ESMC,
)
    # Initialize ED matrices
    EDLInput = convert(Matrix{Float64}, AdjustedEDL[ts:ts+EDHorizon-1, :]')
    EDHAvailInput =
        convert(Matrix{Float64}, params.EDHAvail[ts:ts+EDHorizon-1, :]')
    # EDRAvailInput = convert(
    #     Matrix{Float64},
    #     params.EDRAvail[ts:ts+EDHorizon-1, :]',
    # )
    EDSAvailInput =
        convert(Matrix{Float64}, AdjustedEDSolar[ts:ts+EDHorizon-1, :]')
    EDWAvailInput =
        convert(Matrix{Float64}, AdjustedEDWind[ts:ts+EDHorizon-1, :]')
    EDDBidInput = convert(Matrix{Float64}, RTDBids[:, ts:ts+EDHorizon-1])
    EDCBidInput = convert(Matrix{Float64}, RTCBids[:, ts:ts+EDHorizon-1])

    # Save DAP and RTP for the next day if not the first day
    if d > 1
        last_24_UC = all_UCprices_df[(end-48+h):(end-25+h), :]
        last_36_ED =
            all_EDprices_df[(end-35-BAWindow):end-BAWindow, :] .* EDSteps
        predictors = vcat(last_24_UC, last_36_ED)
    end

    for tp in 1:EDHorizon
        # Ad hoc method to solve initial generation output for ED model
        if ts == 1 && tp == 1 # first day first time period
            for i in axes(params.GPIni, 1)
                set_normalized_rhs(
                    edmodel[:RUIni][i],
                    params.GPIni[i] +
                    params.GRU[i] +
                    (params.GPmin[i]) * EDV[i, (h-1)*EDSteps+t+tp-1],
                )
                set_normalized_rhs(
                    edmodel[:RDIni][i],
                    -params.GPIni[i] +
                    params.GRD[i] +
                    (params.GPmin[i]) * EDW[i, (h-1)*EDSteps+t+tp-1] +
                    (1 - RTOInput[i, tp]) * 10000,
                )
                # set_normalized_rhs(edmodel[:RUIni][i], GPini[i] + GRU[i])
                # set_normalized_rhs(edmodel[:RDIni][i], -GPini[i] + GRD[i])
            end
        elseif h == 1 && t == 1 && tp == 1 # first time period for each day 
            for i in axes(params.GPIni, 1)
                set_normalized_rhs(
                    edmodel[:RUIni][i],
                    EDGPIni[i] +
                    params.GRU[i] / EDSteps +
                    (params.GPmin[i]) * EDV[i, 1],
                )
                set_normalized_rhs(
                    edmodel[:RDIni][i],
                    -EDGPIni[i] +
                    params.GRD[i] / EDSteps +
                    (params.GPmax[i]) * EDW[i, 1] +
                    (1 - RTOInput[i, 1]) * 10000,
                )
                # set_normalized_rhs(edmodel[:RUIni][i], GPini[i] + GRU[i])
                # set_normalized_rhs(edmodel[:RDIni][i], -GPini[i] + GRD[i])
            end
        end
        # update energy storage bids
        if strategic
            # update bids for AI-Enchanced and Baseline energy storage
            # update bids for baseline energy storage using OCB
            for i in axes(RTDBids, 1)
                if params.EStrategic[i] == 0
                    if params.Eeta[i] == 0.9
                        db[i, :] =
                            (
                                vrt[i, :, (h-1)*EDSteps+t] ./ params.Eeta[i] .+
                                ESMC
                            ) / EDSteps
                        db[i, :] = map(
                            x ->
                                x > ESPeakBid ? x * ESPeakBidAdjustment : x,
                            db[i, :],
                        )
                        cb[i, :] =
                            -vrt[i, :, (h-1)*EDSteps+t] .* params.Eeta[i] /
                            EDSteps
                        if tp == 1
                            cbdf = DataFrame(12 * cb[i, :]', :auto)
                            dbdf = DataFrame(12 * db[i, :]', :auto)
                            CSV.write(
                                joinpath(
                                    output_folder * "/NStrategic",
                                    "EDESCB" * "$i" * ".csv",
                                ),
                                cbdf,
                                append = true,
                            )
                            CSV.write(
                                joinpath(
                                    output_folder * "/NStrategic",
                                    "EDESDB" * "$i" * ".csv",
                                ),
                                dbdf,
                                append = true,
                            )
                        end
                    else
                        db[i, :] .= EDDBidInput[i, tp] / EDSteps
                        cb[i, :] .= -EDCBidInput[i, tp] / EDSteps
                    end
                end
            end
            # update bids for AI-Enhaned energy storage using OCB + AI
            for (i, model) in bidmodels
                if d == 1
                    db[i, :] =
                        (vrt[i, :, (h-1)*EDSteps+t] ./ params.Eeta[i] .+ ESMC) /
                        EDSteps
                    db[i, :] = map(
                        x -> x > ESPeakBid ? x * ESPeakBidAdjustment : x,
                        db[i, :],
                    )
                    cb[i, :] =
                        -vrt[i, :, (h-1)*EDSteps+t] .* params.Eeta[i] / EDSteps
                else
                    v = model(
                        predictors[
                            :,
                            findfirst(isequal(1), params.storagemap[i, :]),
                        ],
                    )
                    vavg = [
                        mean(v[(i-1)*segment_length+1:i*segment_length]) for
                        i in 1:ESSeg
                    ]
                    enforce_strictly_decreasing_vector(vavg)
                    cb[i, :] .= -vavg .* 0.9 ./ EDSteps
                    db[i, :] .= (vavg ./ 0.9 .+ ESMC) ./ EDSteps
                    db[i, :] = map(
                        x -> x > ESPeakBid ? x * ESPeakBidAdjustment : x,
                        db[i, :],
                    )
                end
                if tp == 1
                    cbdf = DataFrame(12 * cb[i, :]', :auto)
                    dbdf = DataFrame(12 * db[i, :]', :auto)
                    CSV.write(
                        joinpath(
                            output_folder * "/Strategic",
                            "EDESCB" * "$i" * ".csv",
                        ),
                        cbdf,
                        append = true,
                    )
                    CSV.write(
                        joinpath(
                            output_folder * "/Strategic",
                            "EDESDB" * "$i" * ".csv",
                        ),
                        dbdf,
                        append = true,
                    )
                end
            end
        else
            # update bids for all energy storage using historical bids
            for i in axes(RTDBids, 1)
                db[i, :] .= EDDBidInput[i, tp] / EDSteps
                cb[i, :] .= -EDCBidInput[i, tp] / EDSteps
            end
        end
        # Update new bids in edmodel
        for i in axes(RTDBids, 1)
            for s in 1:ESSeg
                set_objective_coefficient(
                    edmodel,
                    edmodel[:d][i, s, tp],
                    db[i, s],
                )
                set_objective_coefficient(
                    edmodel,
                    edmodel[:c][i, s, tp],
                    cb[i, s],
                )
            end
        end

        for z in axes(EDLInput, 1)
            set_normalized_rhs(edmodel[:LoadBalance][z, tp], EDLInput[z, tp])
        end
        for i in axes(U, 1)
            # set_normalized_rhs(
            #     edmodel[:UCCapU][i, tp],
            #     EDU[i, (h-1)*EDSteps+t+tp-1] * params.GPmax[i],
            # )
            # set_normalized_rhs(
            #     edmodel[:UCCapL][i, tp],
            #     EDU[i, (h-1)*EDSteps+t+tp-1] * params.GPmin[i],
            # )
            # set_normalized_rhs(
            #     edmodel[:UCGenSeg1][i, tp],
            #     EDU[i, (h-1)*EDSteps+t+tp-1] * params.GPmin[i],
            # )
            # TODO Replace this
            set_normalized_rhs(
                edmodel[:UCCapU][i, tp],
                RTOInput[i, tp] *
                EDU[i, (h-1)*EDSteps+t+tp-1] *
                params.GPmax[i],
            )
            set_normalized_rhs(
                edmodel[:UCCapL][i, tp],
                RTOInput[i, tp] *
                EDU[i, (h-1)*EDSteps+t+tp-1] *
                params.GPmin[i],
            )
            set_normalized_rhs(
                edmodel[:UCGenSeg1][i, tp],
                RTOInput[i, tp] *
                EDU[i, (h-1)*EDSteps+t+tp-1] *
                params.GPmin[i],
            )
        end
        for i in axes(EDHAvailInput, 1)
            set_normalized_rhs(edmodel[:HCap][i, tp], EDHAvailInput[i, tp])
        end
        # for i in axes(EDRAvailInput, 1)
        #     set_normalized_rhs(
        #         edmodel[:ReCap][i, tp],
        #         EDRAvailInput[i, tp],
        #     )
        # end
        for z in axes(EDSAvailInput, 1)
            set_normalized_rhs(edmodel[:SCap][z, tp], EDSAvailInput[z, tp])
        end
        for z in axes(EDWAvailInput, 1)
            set_normalized_rhs(edmodel[:WCap][z, tp], EDWAvailInput[z, tp])
        end
        if EDHorizon > 1 && tp > 1
            for i in axes(params.GPIni, 1)
                set_normalized_rhs(
                    edmodel[:RU][i, tp],
                    params.GRU[i] / EDSteps +
                    params.GPmin[i] * EDV[i, (h-1)*EDSteps+t+tp-1],
                )
            end
            for i in axes(params.GPIni, 1)
                # set_normalized_rhs(
                #     edmodel[:RD][i, tp],
                #     params.GRD[i] / EDSteps +
                #     params.GPmax[i] * EDW[i, (h-1)*EDSteps+t+tp-1],
                # )
                # TODO: Replace this
                set_normalized_rhs(
                    edmodel[:RD][i, tp],
                    params.GRD[i] / EDSteps +
                    params.GPmax[i] * EDW[i, (h-1)*EDSteps+t+tp-1] +
                    (1 - RTOInput[i, tp]) * 10000,
                )
            end
        end
    end
end

function writeEDtoCSVandUpdateSOC(
    params,
    output_folder::String,
    edmodel,
    RTO_repeated,
    EDU,
    EDV,
    EDW,
    db,
    cb,
    all_EDprices_df,
    storagezone,
    PriceCap,
    FuelAdjustment,
    EDGSMC,
    EDGMCcost,
    EDGSMCcost,
    EDVOLL,
    EDEScost,
    EDcost,
    EDSteps,
    ESSeg,
    d,
    h,
    t,
)
    ts = ((d - 1) * 24 + h - 1) * EDSteps + t # time step
    EDGMCcost[ts] =
        sum(value.(edmodel[:guc])[:, 1] .* params.GMC * FuelAdjustment) /
        EDSteps
    EDGSMCcost[ts] =
        sum(
            value.(edmodel[:gucs])[:, :, 1] .* EDGSMC[:, :, 1] * FuelAdjustment,
        ) / EDSteps
    EDVOLL[ts] =
        sum(value.(edmodel[:s])[:, :, 1] .* PriceCap[:, :, 1]) / EDSteps
    EDEScost[ts] = sum(
        value.(edmodel[:d])[:, :, 1] .* db[:, :] +
        value.(edmodel[:c])[:, :, 1] .* cb[:, :],
    )
    EDcost[ts] =
        EDGMCcost[ts] +
        EDGSMCcost[ts] +
        EDVOLL[ts] +
        EDEScost[ts] +
        sum(
            EDU[:, (h-1)*EDSteps+t] .* params.GNLC / EDSteps +
            EDV[:, (h-1)*EDSteps+t] .* params.GSUC,
        )
    EDGPIni = value.(edmodel[:guc])[:, 1]
    EDGdf = DataFrame(EDGPIni', :auto)
    CSV.write(joinpath(output_folder, "EDGen.csv"), EDGdf, append = true)
    # EDReGen = value.(edmodel[:gr])[:, 1]
    # EDReGendf = DataFrame(EDReGen', :auto)
    # CSV.write(
    #     joinpath(output_folder, "EDRenewable.csv"),
    #     EDReGendf,
    #     append = true,
    # )
    # EDHydroGen = value.(edmodel[:gh])[:, 1]
    # EDHydroGendf = DataFrame(EDHydroGen', :auto)
    # CSV.write(
    #     joinpath(output_folder, "EDHydro.csv"),
    #     EDHydroGendf,
    #     append = true,
    # )
    # CSV.write(
    #     joinpath(output_folder, "EDDbid.csv"),
    #     DataFrame(EDDBidInput[:, 1]', :auto),
    #     append = true,
    # )
    # CSV.write(
    #     joinpath(output_folder, "EDCbid.csv"),
    #     DataFrame(EDCBidInput[:, 1]', :auto),
    #     append = true,
    # )
    EDESD = value.(edmodel[:totald])[:, 1]
    EDESDdf = DataFrame(EDESD', :auto)
    CSV.write(joinpath(output_folder, "EDESD.csv"), EDESDdf, append = true)
    EDESC = value.(edmodel[:totalc])[:, 1]
    EDESCdf = DataFrame(EDESC', :auto)
    CSV.write(joinpath(output_folder, "EDESC.csv"), EDESCdf, append = true)
    EDSegSOCini = value.(edmodel[:e])[:, :, 1]
    EDSOCini = value.(edmodel[:totale])[:, 1]
    EDSOCinidf = DataFrame(EDSOCini', :auto)
    CSV.write(
        joinpath(output_folder, "EDSOCini.csv"),
        EDSOCinidf,
        append = true,
    )
    EDTrans = value.(edmodel[:f])[:, 1]
    EDTransdf = DataFrame(EDTrans', :auto)
    CSV.write(joinpath(output_folder, "EDTrans.csv"), EDTransdf, append = true)
    EDS = value.(edmodel[:totals])[:, 1]
    EDSdf = DataFrame(EDS', :auto)
    CSV.write(joinpath(output_folder, "EDSlack.csv"), EDSdf, append = true)
    EDprice = dual.(edmodel[:LoadBalance])[:, 1]
    EDpricedf = DataFrame(EDprice', :auto)
    CSV.write(joinpath(output_folder, "EDprice.csv"), EDpricedf, append = true)
    all_EDprices_df = vcat(all_EDprices_df, EDpricedf)
    EDESRev =
        EDprice[storagezone] .*
        (value.(edmodel[:totald])[:, 1] - value.(edmodel[:totalc])[:, 1])
    CSV.write(
        joinpath(output_folder, "EDESRev.csv"),
        DataFrame(EDESRev', :auto),
        append = true,
    )
    EDGZone = DataFrame(value.(edmodel[:guc])[:, 1]' * params.genmap, :auto)
    CSV.write(joinpath(output_folder, "EDGenZone.csv"), EDGZone, append = true)
    # EDHydroZone = DataFrame(
    #     value.(edmodel[:gh])[:, 1]' * params.hydromap,
    #     :auto,
    # )
    # CSV.write(
    #     joinpath(output_folder, "EDHydroZone.csv"),
    #     EDHydroZone,
    #     append = true,
    # )
    EDHydroZone = DataFrame(value.(edmodel[:gh])[:, 1]', :auto)
    CSV.write(
        joinpath(output_folder, "EDHydroZone.csv"),
        EDHydroZone,
        append = true,
    )
    # EDRenewableZone = DataFrame(
    #     value.(edmodel[:gr])[:, 1]' * params.renewablemap,
    #     :auto,
    # )
    # CSV.write(
    #     joinpath(output_folder, "EDRenewableZone.csv"),
    #     EDRenewableZone,
    #     append = true,
    # )
    EDSolarZone = DataFrame(value.(edmodel[:gs])[:, 1]', :auto)
    CSV.write(
        joinpath(output_folder, "EDSolarZone.csv"),
        EDSolarZone,
        append = true,
    )
    EDWindZone = DataFrame(value.(edmodel[:gw])[:, 1]', :auto)
    CSV.write(
        joinpath(output_folder, "EDWindZone.csv"),
        EDWindZone,
        append = true,
    )
    EDESDisZone =
        DataFrame(value.(edmodel[:totald])[:, 1]' * params.storagemap, :auto)
    CSV.write(
        joinpath(output_folder, "EDESDisZone.csv"),
        EDESDisZone,
        append = true,
    )
    EDESChaZone =
        DataFrame(value.(edmodel[:totalc])[:, 1]' * params.storagemap, :auto)
    CSV.write(
        joinpath(output_folder, "EDESChaZone.csv"),
        EDESChaZone,
        append = true,
    )
    EDSOCZone =
        DataFrame(value.(edmodel[:totale])[:, 1]' * params.storagemap, :auto)
    CSV.write(
        joinpath(output_folder, "EDSoCZone.csv"),
        EDSOCZone,
        append = true,
    )
    # Update initial generation output for next time step
    for i in axes(params.GPIni, 1)
        # if (h - 1) * EDSteps + t != UCHorizon * EDSteps
        set_normalized_rhs(
            edmodel[:RUIni][i],
            EDGPIni[i] +
            params.GRU[i] / EDSteps +
            (params.GPmin[i]) * EDV[i, (h-1)*EDSteps+t+1],
        )
        # set_normalized_rhs(
        #     edmodel[:RDIni][i],
        #     -EDGPIni[i] +
        #     params.GRD[i] / EDSteps +
        #     (params.GPmax[i]) * EDW[i, (h-1)*EDSteps+t+1],
        # )
        # TODO: Replace this
        set_normalized_rhs(
            edmodel[:RDIni][i],
            -EDGPIni[i] +
            params.GRD[i] / EDSteps +
            (params.GPmax[i]) * EDW[i, (h-1)*EDSteps+t+1] +
            (1 - RTO_repeated[i, (d-1)*EDSteps*24+(h-1)*EDSteps+t+1]) * 10000,
        )

        # set_normalized_rhs(edmodel[:RUIni][i], EDGPini[i] + GRU[i] + GPmin[i]*EDV[i,(h - 1) * EDSteps + t + 1])
        # set_normalized_rhs(edmodel[:RDIni][i], -EDGPini[i] + GRD[i] + GPmin[i]*EDW[i,(h - 1) * EDSteps + t + 1])
        # end
    end
    for i in axes(EDSOCini, 1)
        for s in 1:ESSeg
            set_normalized_rhs(edmodel[:StorageSOCIni][i, s], EDSegSOCini[i, s])
        end
    end
    return EDGPIni, EDSOCini, all_EDprices_df
end

# Solving unit commitment and economic dispatch model by iterativly updating parameters
function solving(
    params::STESTS.ModelParams,
    Nday::Int,
    strategic::Bool,
    FORB::Bool,
    DADBids::Matrix{Float64},
    DACBids::Matrix{Float64},
    RTDBids::Matrix{Float64},
    RTCBids::Matrix{Float64},
    ucmodel::JuMP.Model,
    ucpmodel::JuMP.Model,
    edmodel::JuMP.Model,
    output_folder::String,
    PriceCap::Array{Float64};
    bidmodels::Vector{Any} = [],
    ESSeg::Int = 1,
    ESMC::Float64 = 10.0,
    UCHorizon::Int = 24,
    EDHorizon::Int = 1,
    EDSteps::Int = 12,
    BAWindow::Int = 0,
    VOLL::Float64 = 9000.0,
    RM::Float64 = 0.03,
    FuelAdjustment::Float64 = 1.0,
    NLCAdjustment::Float64 = 1.0,
    ErrorAdjustment::Float64 = 1.0,
    LoadAdjustment::Float64 = 1.0,
    ESPeakBidAdjustment::Float64 = 1.0,
    ESPeakBid::Float64 = 100.0,
    seed::Int = 0,
)
    GSMC = repeat(params.GSMC, outer = (1, 1, UCHorizon))
    EDGSMC = repeat(params.GSMC, outer = (1, 1, EDHorizon))
    SU = zeros(Int, size(params.GPIni, 1)) # initial generator must on time
    SD = zeros(Int, size(params.GPIni, 1)) # initial generator down on time
    storagezone = [findmax(row)[2] for row in eachrow(params.storagemap)]
    segment_length = 5 ÷ ESSeg

    UCcost = Array{Float64}(undef, Nday)
    GMCcost = Array{Float64}(undef, Nday)
    GSMCcost = Array{Float64}(undef, Nday)
    VOLLcost = Array{Float64}(undef, Nday)
    EScost = Array{Float64}(undef, Nday)
    EDcost = Array{Float64}(undef, Nday * 24 * EDSteps)
    EDGMCcost = Array{Float64}(undef, Nday * 24 * EDSteps)
    EDGSMCcost = Array{Float64}(undef, Nday * 24 * EDSteps)
    EDVOLL = Array{Float64}(undef, Nday * 24 * EDSteps)
    EDEScost = Array{Float64}(undef, Nday * 24 * EDSteps)
    EDGPIni = Array{Float64}(undef, size(params.GPIni, 1))
    GPIniInput = params.GPIni
    all_UCprices_df = DataFrame()
    all_EDprices_df = DataFrame()

    EDprice24 = Array{Float64}(undef, 24, size(params.storagemap, 2))
    vda = Array{Float64}(undef, size(params.storagemap, 1), 24 + 1)
    DAdb = Array{Float64}(undef, size(params.storagemap, 1), 24 + 1)
    DAcb = Array{Float64}(undef, size(params.storagemap, 1), 24 + 1)
    vrt = Array{Float64}(
        undef,
        size(params.storagemap, 1),
        ESSeg,
        24 * EDSteps + 1,
    )

    db = Array{Float64}(undef, size(params.storagemap, 1), ESSeg)
    cb = Array{Float64}(undef, size(params.storagemap, 1), ESSeg)
    AdjustedUCL = params.UCL * LoadAdjustment
    UCL_repeated = repeat(params.UCL, inner = (12, 1))
    LoadError = UCL_repeated - params.EDL
    AdjustedEDL = (UCL_repeated - LoadError * ErrorAdjustment) * LoadAdjustment
    Solar_repeated = repeat(params.SAvail, inner = (12, 1))
    SolarError = Solar_repeated - params.EDSAvail
    AdjustedEDSolar = Solar_repeated - SolarError * ErrorAdjustment
    Wind_repeated = repeat(params.WAvail, inner = (12, 1))
    WindError = Wind_repeated - params.EDWAvail
    AdjustedEDWind = Wind_repeated - WindError * ErrorAdjustment

    if FORB
        RTO = generate_real_time_outage(params.GFOR, params.GOD, seed)
        RTO_repeated = repeat(RTO, inner = (1, EDSteps))
        DAO = generate_day_ahead_outage(RTO)
        df_outage_profile = DataFrame(RTO, :auto)
        CSV.write(joinpath(output_folder, "EDOutage.csv"), df_outage_profile)
        df_day_ahead_profile = DataFrame(DAO, :auto)
        CSV.write(joinpath(output_folder, "UCOutage.csv"), df_day_ahead_profile)
    else
        RTO = ones(Int, size(params.GPIni, 1), 8760)
        RTO_repeated = ones(Int, size(params.GPIni, 1), 8760 * EDSteps)
    end

    set_optimizer_attribute(ucmodel, "MIPGap", 0.001)

    for d in 1:Nday
        # Update load and renewable generation
        LInput = convert(
            Matrix{Float64},
            AdjustedUCL[24*(d-1)+1:24*d+UCHorizon-24, :]',
        )
        HAvailInput = convert(
            Matrix{Float64},
            params.HAvail[24*(d-1)+1:24*d+UCHorizon-24, :]',
        )
        # RAvailInput = convert(
        #     Matrix{Float64},
        #     params.RAvail[24*(d-1)+1:24*d+UCHorizon, :]',
        # )
        SAvailInput = convert(
            Matrix{Float64},
            params.SAvail[24*(d-1)+1:24*d+UCHorizon-24, :]',
        )
        WAvailInput = convert(
            Matrix{Float64},
            params.WAvail[24*(d-1)+1:24*d+UCHorizon-24, :]',
        )
        if FORB
            DAOInput =
                convert(Matrix{Int}, DAO[:, 24*(d-1)+1:24*d+UCHorizon-24])
        else
            DAOInput = ones(Int, size(params.GPIni, 1), UCHorizon)
        end

        if d == 1 || !strategic
            DAdb = convert(
                Matrix{Float64},
                DADBids[:, 24*(d-1)+1:24*d+UCHorizon-24],
            )
            DAcb = convert(
                Matrix{Float64},
                DACBids[:, 24*(d-1)+1:24*d+UCHorizon-24],
            )
        else
            # generate OCB DA bids (non-AI for all ES)
            EDprice288 = all_EDprices_df[(end-287):end, :] .* EDSteps
            for col in 1:size(params.UCL, 2)
                EDprice24[:, col] =
                    mean(reshape(EDprice288[!, col], 12, :), dims = 1)
            end
            for i in axes(DADBids, 1)
                if params.Eeta[i] == 0.9
                    vda[i, :] = generate_value_function(
                        24,
                        params.EPD[i] / params.ESOC[i],
                        params.Eeta[i],
                        1,
                        EDprice24 * params.storagemap[i, :],
                    )
                    DAdb[i, :] = vda[i, :] / params.Eeta[i] .+ ESMC
                    DAcb[i, :] = -vda[i, :] .* params.Eeta[i]
                else
                    DAdb[i, :] = convert(
                        Vector{Float64},
                        DADBids[1, 24*(d-1)+1:24*d+UCHorizon-24],
                    )
                    DAdb[i, :] = map(
                        x -> x > ESPeakBid ? x * ESPeakBidAdjustment : x,
                        DAdb[i, :],
                    )
                    DAcb[i, :] = convert(
                        Vector{Float64},
                        DACBids[1, 24*(d-1)+1:24*d+UCHorizon-24],
                    )
                end
            end
        end

        DAdbdf = DataFrame(DAdb', :auto)
        CSV.write(
            joinpath(output_folder, "UCESDBids.csv"),
            DAdbdf,
            append = true,
        )
        DAcbdf = DataFrame(DAcb', :auto)
        CSV.write(
            joinpath(output_folder, "UCESCBids.csv"),
            DAcbdf,
            append = true,
        )

        # Set unit commitment model constraints
        setUCConstraints(
            params,
            ucmodel,
            ucpmodel,
            LInput,
            HAvailInput,
            WAvailInput,
            SAvailInput,
            DADBids,
            DAdb,
            DAcb,
            FORB,
            DAOInput,
            UCHorizon,
            RM,
        )
        # Solve unit commitment model
        @info "Solving day $d UC."
        optimize!(ucmodel)

        # Extract solution and solve economic dispatch model
        if termination_status(ucmodel) == MOI.OPTIMAL
            GMCcost[d] = sum(
                value.(ucmodel[:guc])[:, 1:24] .* params.GMC * FuelAdjustment,
            )
            GSMCcost[d] = sum(
                value.(ucmodel[:gucs])[:, :, 1:24] .* GSMC[:, :, 1:24] *
                FuelAdjustment,
            )
            VOLLcost[d] = sum(value.(ucmodel[:s])[1:24] .* VOLL)
            EScost[d] = sum(
                value.(ucmodel[:d])[:, 1:24] .* DAdb[:, 1:24] -
                value.(ucmodel[:c])[:, 1:24] .* DAcb[:, 1:24],
            )
            UCcost[d] =
                GMCcost[d] +
                GSMCcost[d] +
                VOLLcost[d] +
                EScost[d] +
                sum(
                    NLCAdjustment * value.(ucmodel[:u])[:, 1:24] .*
                    params.GNLC + value.(ucmodel[:v])[:, 1:24] .* params.GSUC,
                )
            # UCcost[d] = objective_value(ucmodel)
            U = convert(Matrix{Int64}, round.(value.(ucmodel[:u])))
            V = convert(Matrix{Int64}, round.(value.(ucmodel[:v])))
            W = convert(Matrix{Int64}, round.(value.(ucmodel[:w])))
            S = value.(ucmodel[:s])
            Udf = DataFrame(U', :auto)
            Vdf = DataFrame(V', :auto)
            Wdf = DataFrame(W', :auto)
            Sdf = DataFrame(S', :auto)

            writeUCtoCSV(params, output_folder, ucmodel, Udf, Vdf, Wdf, Sdf)

            # If there's an outage end by the end of the day, immediately get the unit back to service
            RTOInput =
                convert(Matrix{Int}, RTO[:, 24*(d-1)+1:24*d+UCHorizon-24])
            for i in axes(V, 1)
                zero_indices = findall(x -> x == 0, RTOInput[i, :])

                for index in zero_indices
                    if index < size(RTOInput, 2) && RTOInput[i, index+1] == 1
                        V[i, index+1] = 1
                    end
                end
            end
            # Update next day's initial generator status allign with real-time outage
            for i in axes(U, 1)
                if RTOInput[i, 24] == 0
                    U[i, 24] = 0
                end
            end

            # Update status and must up/down constraints for next day
            SU = getLastOneIndex(V[:, 1:24], SU, params.GUT)
            SD = getLastOneIndex(W[:, 1:24], SD, params.GDT)
            SUInt = zeros(Int, size(params.GPIni, 1), UCHorizon)
            SDInt = zeros(Int, size(params.GPIni, 1), UCHorizon)

            # If at the begining of the next day, the generator is in outage, then clear remaining must stay on hour
            for i in axes(RTO, 1)
                if RTO[i, 24*d+1] == 0
                    SU[i] = 0
                end
            end

            for i in axes(params.GPIni, 1)
                num_ones_SU = min(SU[i], UCHorizon)
                num_ones_SD = min(SD[i], UCHorizon)
                SUInt[i, 1:num_ones_SU] .= Int64(1)
                SDInt[i, 1:num_ones_SD] .= Int64(1)
            end

            for i in axes(params.GPIni, 1)
                set_normalized_rhs(ucmodel[:ST0][i], U[i, 24])
                set_normalized_rhs(ucmodel[:UTimeIni][i], min(SU[i], UCHorizon))
                set_normalized_rhs(ucmodel[:DTimeIni][i], min(SD[i], UCHorizon))
                for h in 1:UCHorizon
                    set_normalized_coefficient(
                        ucmodel[:UTimeIni][i],
                        ucmodel[:u][i, h],
                        SUInt[i, h],
                    )
                    set_normalized_coefficient(
                        ucmodel[:DTimeIni][i],
                        ucmodel[:u][i, h],
                        SDInt[i, h],
                    )
                end
            end

            calculateDAP(params, ucpmodel, UCHorizon, U, V, W, GPIniInput)

            optimize!(ucpmodel)
            UCprice = dual.(ucpmodel[:LoadBalance][:, 1:24])
            UCpricedf = DataFrame(UCprice', :auto)
            CSV.write(
                joinpath(output_folder, "UCprice.csv"),
                UCpricedf,
                append = true,
            )
            all_UCprices_df = vcat(all_UCprices_df, UCpricedf)
            UCprice288 = repeat(UCprice', inner = (EDSteps, 1))

            # generate non-strategic RT bids
            for i in axes(RTDBids, 1)
                if params.Eeta[i] == 0.9
                    vrt[i, :, :] = generate_value_function(
                        288,
                        params.EPD[i] / params.ESOC[i] / EDSteps,
                        params.Eeta[i],
                        ESSeg,
                        UCprice288 * params.storagemap[i, :],
                    )
                end
            end

            # Pass day-ahead unit commitment results to economic dispatch model, repeat each hour for EDSteps
            EDU = repeat(U, inner = (1, EDSteps))
            EDV = zeros(size(V, 1), size(V, 2) * EDSteps + EDHorizon)
            EDW = zeros(size(V, 1), size(V, 2) * EDSteps + EDHorizon)
            for i in axes(V, 2)
                EDU[:, (i-1)*EDSteps+1] = U[:, i]
                EDV[:, (i-1)*EDSteps+1] = V[:, i]
                EDW[:, (i-1)*EDSteps+1] = W[:, i]
            end

            # Solving economic dispatch model
            for h in 1:24
                for t in 1:EDSteps
                    ts = ((d - 1) * 24 + h - 1) * EDSteps + t # time step
                    RTOInput = convert(
                        Matrix{Float64},
                        RTO_repeated[:, ts:ts+EDHorizon-1],
                    )

                    setEDConstraints(
                        params,
                        output_folder,
                        edmodel,
                        EDV,
                        EDW,
                        EDU,
                        U,
                        EDGPIni,
                        EDSteps,
                        BAWindow,
                        strategic,
                        FORB,
                        bidmodels,
                        db,
                        cb,
                        vrt,
                        segment_length,
                        ESSeg,
                        RTDBids,
                        RTCBids,
                        ESPeakBid,
                        ESPeakBidAdjustment,
                        AdjustedEDL,
                        AdjustedEDSolar,
                        AdjustedEDWind,
                        RTOInput,
                        all_UCprices_df,
                        all_EDprices_df,
                        EDHorizon,
                        d,
                        h,
                        t,
                        ts,
                        ESMC,
                    )

                    # Solve economic dispatch model
                    @info "Solving day $d hour $h step $t ED."
                    optimize!(edmodel)

                    # Extract solution
                    if termination_status(edmodel) == MOI.OPTIMAL
                        EDGPIni, EDSOCini, all_EDprices_df =
                            writeEDtoCSVandUpdateSOC(
                                params,
                                output_folder::String,
                                edmodel,
                                RTO_repeated,
                                EDU,
                                EDV,
                                EDW,
                                db,
                                cb,
                                all_EDprices_df,
                                storagezone,
                                PriceCap,
                                FuelAdjustment,
                                EDGSMC,
                                EDGMCcost,
                                EDGSMCcost,
                                EDVOLL,
                                EDEScost,
                                EDcost,
                                EDSteps,
                                ESSeg,
                                d,
                                h,
                                t,
                            )
                    else
                        solvingError(edmodel, "ED", 0, h, t)
                    end
                    # Update initial generation output for next day
                    if h == 24 && t == 1
                        GPIniInput = EDGPIni
                        for i in axes(params.GPIni, 1)
                            set_normalized_rhs(
                                ucmodel[:RUIni][i],
                                EDGPIni[i] + params.GRU[i],
                            )
                            set_normalized_rhs(
                                ucmodel[:RDIni][i],
                                -EDGPIni[i] + params.GRD[i],
                            )
                            # set_normalized_rhs(
                            #     ucpmodel[:RUIni][i],
                            #     EDGPini[i] + GRU[i],
                            # )
                            # set_normalized_rhs(
                            #     ucpmodel[:RDIni][i],
                            #     -EDGPini[i] + GRD[i],
                            # )
                        end

                        for i in axes(EDSOCini, 1)
                            set_normalized_rhs(
                                ucmodel[:StorageSOCIni][i],
                                EDSOCini[i],
                            )
                            set_normalized_rhs(
                                ucpmodel[:StorageSOCIni][i],
                                EDSOCini[i],
                            )
                        end
                    end
                end
            end
        else
            solvingError(ucmodel, "UC", d, 0, 0)
        end
    end
    EDcostdf = DataFrame(cost = EDcost)
    CSV.write(joinpath(output_folder, "EDcost.csv"), EDcostdf, append = true)
    return UCcost, EDcost
end

function generate_real_time_outage(
    GFOR::Vector{Float64},
    GOD::Vector{Int64},
    seed::Int,
)
    Random.seed!(seed)  # Set the random seed for reproducibility
    num_generators = length(GFOR)
    hours_per_year = 8760
    outage_profile = ones(Int, num_generators, hours_per_year)

    # Generate real-time outage profile
    for i in 1:num_generators
        total_outage_hours = round(Int, hours_per_year * GFOR[i])
        outage_duration = GOD[i]
        remaining_outage_hours = total_outage_hours

        while remaining_outage_hours > 0
            if remaining_outage_hours >= outage_duration
                outage_start = rand(1:(hours_per_year-outage_duration+1))
                outage_end =
                    min(outage_start + outage_duration - 1, hours_per_year)
                actual_outage_duration = outage_end - outage_start + 1

                # Ensure no overlap of outages
                if sum(outage_profile[i, outage_start:outage_end]) ==
                   actual_outage_duration
                    outage_profile[i, outage_start:outage_end] .= 0
                    remaining_outage_hours -= actual_outage_duration
                end
            end

            # If remaining_outage_hours is less than outage_duration, we need to handle it separately
            if remaining_outage_hours > 0 &&
               remaining_outage_hours < outage_duration
                outage_start = rand(1:(hours_per_year-remaining_outage_hours+1))
                outage_end = outage_start + remaining_outage_hours - 1

                if sum(outage_profile[i, outage_start:outage_end]) ==
                   remaining_outage_hours
                    outage_profile[i, outage_start:outage_end] .= 0
                    remaining_outage_hours = 0
                end
            end
        end
    end
    return outage_profile
end

function generate_day_ahead_outage(outage_profile)
    n_generators, n_hours = size(outage_profile)
    hours_per_day = 24
    day_ahead_profile = copy(outage_profile)  # Start with a copy of the real-time profile

    for g in 1:n_generators
        is_outage = false
        start_hour = 0

        for h in 1:n_hours
            if outage_profile[g, h] == 0
                if !is_outage
                    is_outage = true
                    start_hour = h
                    # Calculate the day containing the first zero and set it to all ones in the day-ahead profile
                    start_day = div(start_hour - 1, hours_per_day)
                    day_start_hour = start_day * hours_per_day + 1
                    day_end_hour = (start_day + 1) * hours_per_day
                    day_ahead_profile[
                        g,
                        day_start_hour:min(day_end_hour, n_hours),
                    ] .= 1
                end
            else
                if is_outage
                    is_outage = false
                end
            end
        end
    end

    return day_ahead_profile
end