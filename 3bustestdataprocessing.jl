using JLD2

# D = [100.0 50.0 80.0; 150.0 60.0 100.0; 180.0 80.0 120.0; 100.0 50.0 80.0]
D = [20.0 60.0 50.0; 122.5 60.0 100.0]

transmap = [-1 1 0; -1 0 1; 0 -1 1]
TX = [0.01, 0.01, 0.01]
TFmax = [20.0, 20.0, 20.0]
THurdle = [0.0, 0.0, 0.0]
genmap = [1 0 0; 0 0 1]
GPmax = [100.0, 180.0]
GPmin = [20.0, 50.0]
GNLC = [50.0, 25.0]
GMC = [15.0, 10.0]
GType = ["Gas", "Gas"]
GRU = [45.0, 45.0]
GRD = [45.0, 45.0]
GSUC = [50.0, 50.0]
GUT = [2.0, 2.0]
GDT = [2.0, 2.0]
GPIni = [20.0, 50.0]
hydromap = [0 1 0]
HPmax = [100.0]
HPmin = [0.0]
HAvail = [30.0 50.0 40.0 20.0]
HRamp = [10.0]
renewablemap = [0 1 0]
RPmax = [100.0]
RPmin = [0.0]
RAvail = [30.0 50.0 40.0 20.0]

# if data folder not exist, create a new folder
isdir("data") || mkdir("data")
#  save variables to jld2 file into data folder
@info "Saving data to ./data/3bustest.jld2..."
save(
    "./data/3bustest.jld2",
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
    "D",
    D,
)