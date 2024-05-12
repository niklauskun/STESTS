using CSV
using DataFrames
using JLD2
using Interpolations
# read CSV files

RealTimeNoise = true
Year = 2022
# CurrentMix = true
TransmissionCap = true
DataName = "./data/ADS2032_5GWBES_BS_AggES_" * "$Year" * "_fixed.jld2"
folder = "2032 ADS PCM V2.4.1 Public Data/Processed Data/" * "$Year"

@info "Reading data from $folder..."
timereaddata = @elapsed begin
    # read transmission data
    transmap = Matrix(
        CSV.read(joinpath(folder, "TransmissionMap.csv"), DataFrame)[:, 2:end],
    ) # read transmission line map
    if TransmissionCap == true
        transdata =
            CSV.read(joinpath(folder, "Transmission_Cap.csv"), DataFrame)
    else
        transdata = CSV.read(joinpath(folder, "Transmission.csv"), DataFrame)
    end
    TX = transdata[!, :"X"] # read transmission reactance, in p.u.
    TFmax = transdata[!, :"Max Flow(MW)"] # read transmission line capacity, in MW
    #     THurdle = transdata[!,:"Hurdle Rate($/MW)"] # read transmission line hurdle rate, in $/MW
    @assert length(TX) == length(TFmax) == size(transmap, 1) "Transmission data length mismatch."

    # read generator data
    genmap = Matrix(
        CSV.read(joinpath(folder, "ThermalGenMap_C.csv"), DataFrame)[:, 2:end],
    ) # read generator map
    gendata = CSV.read(joinpath(folder, "ThermalGen_C.csv"), DataFrame)

    GPmax = gendata[!, :"IOMaxCap(MW)"] # read generator maximum capacity, in MW
    GPmin = gendata[!, :"IOMinCap(MW)"] # read generator minimum capacity, in MW
    GMustRun = gendata[!, :"MustRun"] # read generator must run status
    GNLC = gendata[!, :"NoLoadCost(\$)"]  # read generator non-load cost, in $
    GMC = gendata[!, :"VOM Cost"]  # read generator VOM cost, in $/MW
    GSMC = Matrix(gendata[:, 25:29])  # read generator segment marginal cost, in $/MW
    GINCPmax = Matrix(gendata[:, 5:9]) # read generator segment maximum capacity, in MW
    GType = gendata[!, :"SubType"] # read generator type
    GRU = gendata[!, :"RampUp Rate(MW/minute)"] * 60 # read generator ramp up rate, in MW/hour
    GRD = gendata[!, :"RampDn Rate(MW/minute)"] * 60 # read generator ramp down rate, in MW/hour
    GSUC = gendata[!, :"Start Up Cost(\$)"] # read generator start-up cost, in $
    GUT = gendata[!, :"MinimumUpTime(hr)"] # read generator minimum up time, in hour
    GDT = gendata[!, :"MinimumDownTime(hr)"] # read generator minimum down time, in hour
    GPIni = gendata[!, :"InitialDispatch(MW)"] # read generator initial output
    @assert length(GPmax) ==
            length(GPmin) ==
            length(GMustRun) ==
            length(GNLC) ==
            length(GMC) ==
            length(GType) ==
            length(GRU) ==
            length(GRD) ==
            length(GSUC) ==
            length(GUT) ==
            length(GDT) ==
            length(GPIni) ==
            size(GSMC, 1) ==
            size(GINCPmax, 1) ==
            size(genmap, 1) "Generator data length mismatch."

    # read renewable data
    # solarmap = Matrix(
    #     CSV.read(joinpath(folder, "SolarMap_C.csv"), DataFrame)[:, 2:end],
    # )
    # windmap = Matrix(
    #     CSV.read(joinpath(folder, "WindMap_C.csv"), DataFrame)[:, 2:end],
    # )
    # solarprofile = Matrix(
    #     CSV.read(joinpath(folder, "SolarProfile_C.csv"), DataFrame)[:, 2:end],
    # )
    # windprofile = Matrix(
    #     CSV.read(joinpath(folder, "WindProfile_C.csv"), DataFrame)[:, 2:end],
    # )
    # solardata = CSV.read(joinpath(folder, "Solar_C.csv"), DataFrame)
    # winddata = CSV.read(joinpath(folder, "Wind_C.csv"), DataFrame)

    # solarPmax = solardata[!, :"Capacity(MW)"] # read solar maximum capacity, in MW
    # windPmax = winddata[!, :"Capacity(MW)"] # read wind maximum capacity, in MW
    # renewablemap = vcat(solarmap, windmap)
    # RPmax = vcat(solarPmax, windPmax)
    # RAvail = hcat(solarprofile, windprofile)
    # renewablemap = XLSX.readdata(filename, "TotalRenewable", "G3:IL61") # read renewable map
    # RPmax = vec(XLSX.readdata(filename, "TotalRenewable!B3:B61")) # read renewable maximum capacity, in MW
    # RPmin = vec(XLSX.readdata(filename, "TotalRenewable!D3:D61")) # read renewable minimum capacity, in MW
    # RAvail = XLSX.readdata(filename, "TotalRenewable!JD3:LJ8786") # read renewable availability, in MW
    # # RAvail value less than RPmin, set to RPmin, value greater than RPmax, set to RPmax
    # for columns in axes(RAvail, 2)
    #     for rows in axes(RAvail, 1)
    #         if RAvail[rows, columns] < RPmin[columns]
    #             RAvail[rows, columns] = RPmin[columns]
    #             @info "Renewable availability less than minimum capacity at row $rows, column $columns."
    #         elseif RAvail[rows, columns] > RPmax[columns]
    #             RAvail[rows, columns] = RPmax[columns]
    #             @info "Renewable availability less than minimum capacity at row $rows, column $columns."
    #         end
    #     end
    # end
    # @assert length(RPmax) == size(renewablemap, 1) == size(RAvail, 2) "Renewable data length mismatch."

    SAvail = Matrix(
        CSV.read(joinpath(folder, "SolarZone_C.csv"), DataFrame)[:, 2:end],
    )
    WAvail = Matrix(
        CSV.read(joinpath(folder, "WindZone_C.csv"), DataFrame)[:, 2:end],
    )
    HAvail = Matrix(
        CSV.read(joinpath(folder, "HydroZone_C.csv"), DataFrame)[:, 2:end],
    )

    # read hydro region Pmin data
    # TODO: Get Region Pmin!
    # hydromap = Matrix(
    #     CSV.read(joinpath(folder, "HydroMap_C.csv"), DataFrame)[:, 2:end],
    # ) # read hydro map
    hydrodata = CSV.read(joinpath(folder, "HydroZoneCap_C.csv"), DataFrame)
    # HAvail = Matrix(
    #     CSV.read(joinpath(folder, "HydroProfile_C.csv"), DataFrame)[:, 2:end],
    # ) # read hydro availability, in MW

    HPmax = hydrodata[!, :"MaxCap(MW)"] # read hydro maximum capacity, in MW    
    HPmin = hydrodata[!, :"MinCap(MW)"] # read hydro minimum capacity, in MW
    # HAvail value less than HPmin, set to HPmin, value greater than HPmax, set to HPmax
    #     for columns in axes(HAvail, 2)
    #         for rows in axes(HAvail, 1)
    #             if HAvail[rows, columns] < HPmin[columns]
    #                 HAvail[rows, columns] = HPmin[columns]
    #                 @info "Hydro availability less than minimum capacity at row $rows, column $columns."
    #             elseif HAvail[rows, columns] > HPmax[columns]
    #                 HAvail[rows, columns] = HPmax[columns]
    #                 @info "Hydro availability less than minimum capacity at row $rows, column $columns."
    #             end
    #         end
    #     end
    #     HRamp = vec(XLSX.readdata(filename, "HydroRamp!B3:B29")) # read hydro ramp rate, in MW/hour
    @assert length(HPmax) ==
            length(HPmin) ==
            # size(hydromap, 1) ==
            size(HAvail, 2) "Hydro data length mismatch."

    # read storage data
    storagemap = Matrix(
        CSV.read(joinpath(folder, "StorageMap_C_Agg.csv"), DataFrame)[:, 2:end],
    )
    storagedata = CSV.read(
        joinpath(folder, "Storage_C_4hr_5GW_Strategic_Agg.csv"),
        DataFrame,
    )
    ESDABids =
        Matrix(CSV.read(joinpath(folder, "StorageDABids.csv"), DataFrame))
    ESRTBids =
        Matrix(CSV.read(joinpath(folder, "StorageRTBids.csv"), DataFrame))

    EPC = -storagedata[!, :"MinCap(MW)"] # read storage charge capacity, in MW
    EPD = storagedata[!, :"MaxCap(MW)"] # read storage discharge capacity, in MW
    Eeta = storagedata[!, :"Efficiency"] # read storage efficiency
    ESOC = storagedata[!, :"MaxCap(MWh)"] # read storage state of charge capacity, in MWh
    ESOCini = 0.5 * ESOC # initial state of charge, in MWh
    EStrategic = storagedata[!, :"Strategic"] # read storage strategic status
    @assert length(EPC) == size(storagemap, 1) "storage data length mismatch."

    # read demand data
    UCL = Matrix(CSV.read(joinpath(folder, "Load_C.csv"), DataFrame)[:, 2:end]) # read demand, in MW

    @assert size(transmap, 2) ==
            size(genmap, 2) ==
            # size(hydromap, 2) ==
            size(UCL, 2) "Bus number mismatch."
    @assert size(HAvail, 1) == size(UCL, 1) "Time step mismatch."

    #     # replace missing data to 0
    #     transmap = replace(transmap, missing => 0)
    #     genmap = replace(genmap, missing => 0)
    #     hydromap = replace(hydromap, missing => 0)
    #     renewablemap = replace(renewablemap, missing => 0)

    #     # convert data to Julia data type
    transmap = convert(Matrix{Int64}, transmap)
    TX = convert(Vector{Float64}, TX)
    TFmax = convert(Vector{Float64}, TFmax)
    # THurdle = convert(Vector{Float64}, THurdle)
    genmap = convert(Matrix{Int64}, genmap)
    GPmax = convert(Vector{Float64}, GPmax)
    GPmin = convert(Vector{Float64}, GPmin)
    GMustRun = convert(Vector{Int64}, GMustRun)
    GNLC = convert(Vector{Float64}, GNLC)
    GMC = convert(Vector{Float64}, GMC)
    GSMC = convert(Matrix{Float64}, GSMC)
    GINCPmax = convert(Matrix{Float64}, GINCPmax)
    GType = convert(Vector{String}, GType)
    GRU = convert(Vector{Float64}, GRU)
    GRD = convert(Vector{Float64}, GRD)
    GSUC = convert(Vector{Float64}, GSUC)
    GUT = convert(Vector{Int64}, GUT)
    GDT = convert(Vector{Int64}, GDT)
    GPIni = convert(Vector{Float64}, GPIni)
    # hydromap = convert(Matrix{Int64}, hydromap)
    HPmax = convert(Vector{Float64}, HPmax)
    HPmin = convert(Vector{Float64}, HPmin)
    HAvail = convert(Matrix{Float64}, HAvail)
    # HRamp = convert(Vector{Float64}, HRamp)
    # renewablemap = convert(Matrix{Int64}, renewablemap)
    # RPmax = convert(Vector{Float64}, RPmax)
    # RPmin = convert(Vector{Float64}, RPmin)
    # RAvail = convert(Matrix{Float64}, RAvail)
    SAvail = convert(Matrix{Float64}, SAvail)
    WAvail = convert(Matrix{Float64}, WAvail)
    storagemap = convert(Matrix{Int64}, storagemap)
    EPC = convert(Vector{Float64}, EPC)
    EPD = convert(Vector{Float64}, EPD)
    Eeta = convert(Vector{Float64}, Eeta)
    ESOC = convert(Vector{Float64}, ESOC)
    ESOCini = convert(Vector{Float64}, ESOCini)
    EStrategic = convert(Vector{Int64}, EStrategic)
    UCL = convert(Matrix{Float64}, UCL)

    if RealTimeNoise == true
        EDL = Matrix(
            CSV.read(
                joinpath(folder, "realtimeload_all.csv"),
                DataFrame,
                header = false,
            )[
                :,
                :,
            ],
        ) # read demand, in MW
        EDSAvail = Matrix(
            CSV.read(
                joinpath(folder, "realtimesolar_all.csv"),
                DataFrame,
                header = false,
            )[
                :,
                :,
            ],
        )
        EDWAvail = Matrix(
            CSV.read(
                joinpath(folder, "realtimewind_all.csv"),
                DataFrame,
                header = false,
            )[
                :,
                :,
            ],
        )
    else
        EDL = zeros(105408, size(UCL, 2))
        for i in axes(UCL, 2)
            # Create an interpolation object for the current column
            itp = interpolate(UCL[:, i], BSpline(Linear()))

            # Scale the interpolation to the desired number of points
            scale = length(itp) / size(EDL, 1)

            # Fill the new matrix with the interpolated data
            for j in 1:size(EDL, 1)-12+1
                EDL[j, i] = itp((j - 1) * scale + 1)
            end
        end
        # EDRAvail = zeros(105408, size(RAvail, 2))
        EDSAvail = zeros(105408, size(SAvail, 2))
        EDWAvail = zeros(105408, size(WAvail, 2))

        # for i in axes(RAvail, 2)
        #     # Create an interpolation object for the current column
        #     itp = interpolate(RAvail[:, i], BSpline(Linear()))

        #     # Scale the interpolation to the desired number of points
        #     scale = length(itp) / size(EDRAvail, 1)

        #     # Fill the new matrix with the interpolated data
        #     for j in 1:size(EDRAvail, 1)-12+1
        #         EDRAvail[j, i] = itp((j - 1) * scale + 1)
        #     end
        # end

        for i in axes(SAvail, 2)
            # Create an interpolation object for the current column
            itp = interpolate(SAvail[:, i], BSpline(Linear()))

            # Scale the interpolation to the desired number of points
            scale = length(itp) / size(EDSAvail, 1)

            # Fill the new matrix with the interpolated data
            for j in 1:size(EDSAvail, 1)-12+1
                EDSAvail[j, i] = itp((j - 1) * scale + 1)
            end
        end

        for i in axes(WAvail, 2)
            # Create an interpolation object for the current column
            itp = interpolate(WAvail[:, i], BSpline(Linear()))

            # Scale the interpolation to the desired number of points
            scale = length(itp) / size(EDWAvail, 1)

            # Fill the new matrix with the interpolated data
            for j in 1:size(EDWAvail, 1)-12+1
                EDWAvail[j, i] = itp((j - 1) * scale + 1)
            end
        end
    end

    EDHAvail = zeros(105408, size(HAvail, 2))
    for i in axes(HAvail, 2)
        # Create an interpolation object for the current column
        itp = interpolate(HAvail[:, i], BSpline(Linear()))

        # Scale the interpolation to the desired number of points
        scale = length(itp) / size(EDHAvail, 1)

        # Fill the new matrix with the interpolated data
        for j in 1:size(EDHAvail, 1)-12+1
            EDHAvail[j, i] = itp((j - 1) * scale + 1)
        end
    end

    # if data folder not exist, create a new folder
    isdir("data") || mkdir("data")
    #  save variables to jld2 file into data folder
    @info "Saving data to ./data/WECC240.jld2..."
    save(
        DataName,
        "transmap",
        transmap,
        "TX",
        TX,
        "TFmax",
        TFmax,
        # "THurdle",
        # THurdle,
        "genmap",
        genmap,
        "GPmax",
        GPmax,
        "GPmin",
        GPmin,
        "GMustRun",
        GMustRun,
        "GNLC",
        GNLC,
        "GMC",
        GMC,
        "GSMC",
        GSMC,
        "GINCPmax",
        GINCPmax,
        "GType",
        GType,
        "GRU",
        GRU,
        "GRD",
        GRD,
        "GSUC",
        GSUC,
        "GUT",
        GUT,
        "GDT",
        GDT,
        "GPIni",
        GPIni,
        # "hydromap",
        # hydromap,
        "HPmax",
        HPmax,
        "HPmin",
        HPmin,
        "HAvail",
        HAvail,
        # "HRamp",
        # HRamp,
        # "renewablemap",
        # renewablemap,
        # "RPmax",
        # RPmax,
        # "RPmin",
        # RPmin,
        # "RAvail",
        # RAvail,
        "SAvail",
        SAvail,
        "WAvail",
        WAvail,
        "storagemap",
        storagemap,
        "EPC",
        EPC,
        "EPD",
        EPD,
        "Eeta",
        Eeta,
        "ESOC",
        ESOC,
        "ESOCini",
        ESOCini,
        "EStrategic",
        EStrategic,
        "ESDABids",
        ESDABids,
        "ESRTBids",
        ESRTBids,
        "UCL",
        UCL,
        "EDL",
        EDL,
        "EDHAvail",
        EDHAvail,
        # "EDRAvail",
        # EDRAvail,
        "EDSAvail",
        EDSAvail,
        "EDWAvail",
        EDWAvail,
    )
end
@info "Reading data took $timereaddata seconds."
