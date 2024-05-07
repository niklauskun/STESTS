module STESTS

# export read_jld2, unit_commitment, economic_dispatch, solving

include("read.jl") # Read data from .jld2 file and check input data format. (Nik)
include("bidding.jl") # Generate bidding curves for each energy storage unit. (Xin)
# include("realtimenoise.jl") # Generate real-time noise for renewable and load. (Teliang)
include("unitcommitment.jl") # Formulate unit commitment model base on input data. (Nik)
include("unitcommitmentprice.jl") # Formulate unit commitment model base on input data. (Nik)
include("economicdispatch.jl") # Formulate economic dispatch model base on input data. (Nik)
include("solving.jl") # Solve unit commitment and economic dispatch model. (Nik)

end # module STESTS
