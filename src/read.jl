using JLD2
using DataFrames

mutable struct ModelParams
    UCL::Matrix{Float64}
    genmap::Matrix{Int64}
    GPmax::Vector{Float64}
    GPmin::Vector{Float64}
    GMustRun::Vector{Int64}
    GMC::Vector{Float64}
    GSMC::Matrix{Float64}
    GINCPmax::Matrix{Float64}
    transmap::Matrix{Int64}
    TX::Vector{Float64}
    TFmax::Vector{Float64}
    GNLC::Vector{Float64}
    GRU::Vector{Float64}
    GRD::Vector{Float64}
    GSUC::Vector{Float64}
    GUT::Vector{Int64}
    GDT::Vector{Int64}
    GPIni::Vector{Float64}
    GFOR::Vector{Float64}
    GOD::Vector{Int64}
    # hydromap::Matrix{Int64}
    HPmin::Vector{Float64}
    HAvail::Matrix{Float64}
    # renewablemap::Matrix{Int64}
    # RAvail::Matrix{Float64}
    SAvail::Matrix{Float64}
    WAvail::Matrix{Float64}
    storagemap::Matrix{Int64}
    EPC::Vector{Float64}
    EPD::Vector{Float64}
    Eeta::Vector{Float64}
    ESOC::Vector{Float64}
    ESOCini::Vector{Float64}
    EStrategic::Vector{Int64}
    EMC::Vector{Float64}
    ESDABids::Matrix{Float64}
    ESRTBids::Matrix{Float64}
    EDL::Matrix{Float64}
    EDHAvail::Matrix{Float64}
    # EDRAvail::Matrix{Float64}
    EDSAvail::Matrix{Float64}
    EDWAvail::Matrix{Float64}
end

"""
    read_jld2(filename::String)

Read system parameters and save into individual variables from a .jld2 file. 
Variables could be directly used in the unit commitment and economic dispatch construction.

# Example
```julia
read_jld2("./data/WECC240.jld2")
```
"""
function read_jld2(filename::String)
    @info "Reading data from $filename..."
    data = load(filename)
    UCL = data["UCL"]
    genmap = data["genmap"]
    GPmax = data["GPmax"]
    GPmin = data["GPmin"]
    GMustRun = data["GMustRun"]
    GMC = data["GMC"]
    GSMC = data["GSMC"]
    GINCPmax = data["GINCPmax"]
    transmap = data["transmap"]
    TX = data["TX"]
    TFmax = data["TFmax"]
    GNLC = data["GNLC"]
    GRU = data["GRU"]
    GRD = data["GRD"]
    GSUC = data["GSUC"]
    GUT = data["GUT"]
    GDT = data["GDT"]
    GPIni = data["GPIni"]
    GFOR = data["GFOR"]
    GOD = data["GOD"]
    # hydromap = data["hydromap"]
    HPmin = data["HPmin"]
    HAvail = data["HAvail"]
    # renewablemap = data["renewablemap"]
    # RAvail = data["RAvail"]
    SAvail = data["SAvail"]
    WAvail = data["WAvail"]
    storagemap = data["storagemap"]
    EPC = data["EPC"]
    EPD = data["EPD"]
    Eeta = data["Eeta"]
    ESOC = data["ESOC"]
    ESOCini = data["ESOCini"]
    EStrategic = data["EStrategic"]
    EMC = data["EMC"]
    ESDABids = data["ESDABids"]
    ESRTBids = data["ESRTBids"]
    EDL = data["EDL"]
    EDHAvail = data["EDHAvail"]
    # EDRAvail = data["EDRAvail"]
    EDSAvail = data["EDSAvail"]
    EDWAvail = data["EDWAvail"]
    params = ModelParams(
        UCL,
        genmap,
        GPmax,
        GPmin,
        GMustRun,
        GMC,
        GSMC,
        GINCPmax,
        transmap,
        TX,
        TFmax,
        GNLC,
        GRU,
        GRD,
        GSUC,
        GUT,
        GDT,
        GPIni,
        GFOR,
        GOD,
        # hydromap,
        HPmin,
        HAvail,
        # renewablemap,
        # RAvail,
        SAvail,
        WAvail,
        storagemap,
        EPC,
        EPD,
        Eeta,
        ESOC,
        ESOCini,
        EStrategic,
        EMC,
        ESDABids,
        ESRTBids,
        EDL,
        EDHAvail,
        # EDRAvail,
        EDSAvail,
        EDWAvail,
    )
    @info "Done"
    return params
end
