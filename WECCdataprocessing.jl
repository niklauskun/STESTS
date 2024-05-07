using XLSX
using DataFrames
using JLD2
using Interpolations
# read excel file in multiple books and return a dictionary of DataFrames
filename = "./Input_Data_WECC240.xlsx"
@info "Reading data from $filename..."
timereaddata = @elapsed begin
    # read transmission data
    transmap = XLSX.readdata(filename, "Line_Map!H4:IM451") # read transmission line map
    TX = vec(XLSX.readdata(filename, "Line_Data!B3:B450")) # read transmission reactance, in p.u.
    TFmax = vec(XLSX.readdata(filename, "Line_Data", "F3:F450";)) # read transmission line capacity, in MW
    THurdle = vec(XLSX.readdata(filename, "Line_Data!H3:H450")) # read transmission line hurdle rate, in $/MW
    @assert length(TX) == length(TFmax) == length(THurdle) == size(transmap, 1) "Transmission data length mismatch."

    # read generator data
    genmap = XLSX.readdata(filename, "Generator_Map!H4:IM74") # read generator map
    GPmax = vec(XLSX.readdata(filename, "Generator_Data!B3:B73")) # read generator maximum capacity, in MW
    GPmin = vec(XLSX.readdata(filename, "Generator_Data!D3:D73")) # read generator minimum capacity, in MW
    GNLC = vec(XLSX.readdata(filename, "Generator_Data!H3:H73")) # read generator non-load cost, in $
    GMC = vec(XLSX.readdata(filename, "Generator_Data!J3:J73")) # read generator marginal cost, in $/MWh
    GType = vec(XLSX.readdata(filename, "Generator_Data!L3:L73")) # read generator type
    GRU = vec(XLSX.readdata(filename, "Generator_Data!N3:N73")) # read generator ramp up rate, in MW/hour
    GRD = vec(XLSX.readdata(filename, "Generator_Data!P3:P73")) # read generator ramp down rate, in MW/hour
    GSUC = vec(XLSX.readdata(filename, "Generator_Data!R3:R73")) # read generator start-up cost, in $
    GUT = vec(XLSX.readdata(filename, "Generator_Data!T3:T73")) # read generator minimum up time, in hour
    GDT = vec(XLSX.readdata(filename, "Generator_Data!V3:V73")) # read generator minimum down time, in hour
    GPIni = vec(XLSX.readdata(filename, "Generator_Data!AB3:AB73")) # read generator initial output
    @assert length(GPmax) ==
            length(GPmin) ==
            length(GNLC) ==
            length(GMC) ==
            length(GType) ==
            length(GRU) ==
            length(GRD) ==
            length(GSUC) ==
            length(GUT) ==
            length(GDT) ==
            length(GPIni) ==
            size(genmap, 1) "Generator data length mismatch."

    # read hydro data
    hydromap = XLSX.readdata(filename, "Hydro!G3:IL29") # read hydro map
    #transpose of hydromap
    HPmax = vec(XLSX.readdata(filename, "Hydro!B3:B29")) # read hydro maximum capacity, in MW    
    # read xlsx a column save to vector    
    HPmin = vec(XLSX.readdata(filename, "Hydro!D3:D29")) # read hydro minimum capacity, in MW
    HAvail = XLSX.readdata(filename, "Hydro!JD3:KD8786") # read hydro availability, in MW
    # HAvail value less than HPmin, set to HPmin, value greater than HPmax, set to HPmax
    for columns in axes(HAvail, 2)
        for rows in axes(HAvail, 1)
            if HAvail[rows, columns] < HPmin[columns]
                HAvail[rows, columns] = HPmin[columns]
                @info "Hydro availability less than minimum capacity at row $rows, column $columns."
            elseif HAvail[rows, columns] > HPmax[columns]
                HAvail[rows, columns] = HPmax[columns]
                @info "Hydro availability less than minimum capacity at row $rows, column $columns."
            end
        end
    end
    HRamp = vec(XLSX.readdata(filename, "HydroRamp!B3:B29")) # read hydro ramp rate, in MW/hour
    @assert length(HPmax) == length(HPmin) == length(HRamp) == size(hydromap, 1) "Hydro data length mismatch."

    # read renewable data
    renewablemap = XLSX.readdata(filename, "TotalRenewable", "G3:IL61") # read renewable map
    RPmax = vec(XLSX.readdata(filename, "TotalRenewable!B3:B61")) # read renewable maximum capacity, in MW
    RPmin = vec(XLSX.readdata(filename, "TotalRenewable!D3:D61")) # read renewable minimum capacity, in MW
    RAvail = XLSX.readdata(filename, "TotalRenewable!JD3:LJ8786") # read renewable availability, in MW
    # RAvail value less than RPmin, set to RPmin, value greater than RPmax, set to RPmax
    for columns in axes(RAvail, 2)
        for rows in axes(RAvail, 1)
            if RAvail[rows, columns] < RPmin[columns]
                RAvail[rows, columns] = RPmin[columns]
                @info "Renewable availability less than minimum capacity at row $rows, column $columns."
            elseif RAvail[rows, columns] > RPmax[columns]
                RAvail[rows, columns] = RPmax[columns]
                @info "Renewable availability less than minimum capacity at row $rows, column $columns."
            end
        end
    end
    @assert length(RPmax) == length(RPmin) == size(renewablemap, 1) "Renewable data length mismatch."

    # read demand data
    UCL = XLSX.readdata(filename, "Load!D3:II8786") # read demand, in MW
    @assert size(transmap, 2) ==
            size(genmap, 2) ==
            size(hydromap, 2) ==
            size(renewablemap, 2) ==
            size(UCL, 2) "Bus number mismatch."
    @assert size(HAvail, 1) == size(RAvail, 1) == size(UCL, 1) "Time step mismatch."

    # replace missing data to 0
    transmap = replace(transmap, missing => 0)
    genmap = replace(genmap, missing => 0)
    hydromap = replace(hydromap, missing => 0)
    renewablemap = replace(renewablemap, missing => 0)

    # convert data to Julia data type
    transmap = convert(Matrix{Int64}, transmap)
    TX = convert(Vector{Float64}, TX)
    TFmax = convert(Vector{Float64}, TFmax)
    THurdle = convert(Vector{Float64}, THurdle)
    genmap = convert(Matrix{Int64}, genmap)
    GPmax = convert(Vector{Float64}, GPmax)
    GPmin = convert(Vector{Float64}, GPmin)
    GNLC = convert(Vector{Float64}, GNLC)
    GMC = convert(Vector{Float64}, GMC)
    GType = convert(Vector{String}, GType)
    GRU = convert(Vector{Float64}, GRU)
    GRD = convert(Vector{Float64}, GRD)
    GSUC = convert(Vector{Float64}, GSUC)
    GUT = convert(Vector{Int64}, GUT)
    GDT = convert(Vector{Int64}, GDT)
    GPIni = convert(Vector{Float64}, GPIni)
    hydromap = convert(Matrix{Int64}, hydromap)
    HPmax = convert(Vector{Float64}, HPmax)
    HPmin = convert(Vector{Float64}, HPmin)
    HAvail = convert(Matrix{Float64}, HAvail)
    HRamp = convert(Vector{Float64}, HRamp)
    renewablemap = convert(Matrix{Int64}, renewablemap)
    RPmax = convert(Vector{Float64}, RPmax)
    RPmin = convert(Vector{Float64}, RPmin)
    RAvail = convert(Matrix{Float64}, RAvail)
    UCL = convert(Matrix{Float64}, UCL)

    EDL = zeros(105408, size(UCL, 2))
    EDHAvail = zeros(105408, size(HAvail, 2))
    EDRAvail = zeros(105408, size(RAvail, 2))

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

    for i in axes(RAvail, 2)
        # Create an interpolation object for the current column
        itp = interpolate(RAvail[:, i], BSpline(Linear()))

        # Scale the interpolation to the desired number of points
        scale = length(itp) / size(EDRAvail, 1)

        # Fill the new matrix with the interpolated data
        for j in 1:size(EDRAvail, 1)-12+1
            EDRAvail[j, i] = itp((j - 1) * scale + 1)
        end
    end

    # if data folder not exist, create a new folder
    isdir("data") || mkdir("data")
    #  save variables to jld2 file into data folder
    @info "Saving data to ./data/WECC240.jld2..."
    save(
        "./data/WECC240.jld2",
        "transmap",
        transmap,
        "TX",
        TX,
        "TFmax",
        TFmax,
        "THurdle",
        THurdle,
        "genmap",
        genmap,
        "GPmax",
        GPmax,
        "GPmin",
        GPmin,
        "GNLC",
        GNLC,
        "GMC",
        GMC,
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
        "hydromap",
        hydromap,
        "HPmax",
        HPmax,
        "HPmin",
        HPmin,
        "HAvail",
        HAvail,
        "HRamp",
        HRamp,
        "renewablemap",
        renewablemap,
        "RPmax",
        RPmax,
        "RPmin",
        RPmin,
        "RAvail",
        RAvail,
        "UCL",
        UCL,
        "EDL",
        EDL,
        "EDHAvail",
        EDHAvail,
        "EDRAvail",
        EDRAvail,
    )
end
@info "Reading data took $timereaddata seconds."
