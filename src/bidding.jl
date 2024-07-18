# generate energy storage bid using deep learning

using Flux, JLD2, Random

# function loadbidmodels(
#     model_filenames;
#     input_size = 60,
#     dense_size = 60,
#     output_size = 5,
#     activation_fn = relu,
# )
#     models = Dict{String,Chain}()

#     for filename in model_filenames
#         jldopen(filename, "r") do file
#             # Define the model architecture and create a model
# model = Chain(
#     Dense(input_size, dense_size, activation_fn),
#     Dense(dense_size, dense_size, activation_fn),
#     Dense(dense_size, output_size),
# )

#             # Load the model state from the JLD2 file
#             model_state = file["model_state"]
#             Flux.loadmodel!(model, model_state)

#             # Store the loaded model in the dictionary with filename as key
#             return models[filename] = model
#         end
#     end

#     return models
# end

function loadbidmodels(
    base_folder;
    input_size = 60,
    dense_size = 60,
    output_size = 5,
    activation_fn = relu,
)
    models = Dict{String,Dict{String,Chain}}()

    for region in 1:6  # Assuming Region7 has no storage
        region_folder = joinpath(base_folder, "Region$(region)")
        region_models = Dict{String,Chain}()

        for filename in readdir(region_folder, join = true)
            if occursin(r"\.jld2$", filename)  # Ensure it's a JLD2 file
                jldopen(filename, "r") do file
                    model = Chain(
                        Dense(input_size, dense_size, activation_fn),
                        Dense(dense_size, dense_size, activation_fn),
                        Dense(dense_size, output_size),
                    )

                    # Load the model state from the JLD2 file
                    model_state = file["model_state"]
                    Flux.loadmodel!(model, model_state)

                    return region_models[basename(filename)] = model
                end
            end
        end

        models["Region$(region)"] = region_models
    end

    return models
end

# function assign_models_to_storages(params, models, num_storages, output_folder)
#     storage_to_model_map = []
#     storage_to_index_map = []

#     model_keys = collect(keys(models))  # Get the keys of the models dictionary to choose from

#     for storage_id in 1:num_storages
#         if params.EStrategic[storage_id] == 1
#             # Select a model for this storage in a round-robin fashion
#             selected_model_index = ((storage_id - 1) % length(model_keys)) + 1
#             selected_model_key = model_keys[selected_model_index]
#             push!(
#                 storage_to_model_map,
#                 (storage_id, models[selected_model_key]),
#             )
#             push!(storage_to_index_map, (storage_id, selected_model_index)) # Track storage_id and model index
#         end
#     end
#     df = DataFrame(
#         StorageID = first.(storage_to_index_map),
#         SelectedModelIndex = last.(storage_to_index_map),
#     )
#     CSV.write(
#         joinpath(output_folder * "/Strategic", "storage_to_index_map.csv"),
#         df,
#     )

#     return storage_to_model_map
# end

function assign_models_to_storages(
    params,
    models,
    num_storages,
    output_folder;
    RandomModel = false,
    RandomSeed = 0,
)
    storage_to_model_map = []
    storage_to_index_map = []

    # Initialize a dictionary to keep track of the last model index used for each region
    last_model_index_used = Dict{String,Int}()
    unused_model_indices = Dict{String,Vector{Int}}()

    # Set the random seed if RandomModel is true
    if RandomModel
        Random.seed!(RandomSeed)
    end

    for storage_id in 1:num_storages
        if params.EStrategic[storage_id] == 1 ||
           params.EStrategic[storage_id] == 2
            region_idx =
                findfirst(x -> x == 1, params.storagemap[storage_id, 1:6])
            if region_idx !== nothing
                region_key = "Region$(region_idx)"

                # Determine the duration of the storage and select the appropriate model set
                duration = params.ESOC[storage_id] / params.EPC[storage_id]
                efficiency = params.Eeta[storage_id]
                model_prefix = "$(Int(round(duration)))hr$(efficiency)model"

                model_keys = filter(
                    key -> startswith(key, model_prefix),
                    collect(keys(models[region_key])),
                )
                num_models = length(model_keys)
                if num_models == 0
                    error(
                        "No models found for duration $(duration) hours and efficiency $(efficiency) in region $(region_idx)",
                    )
                end

                if RandomModel
                    # Initialize or replenish the unused_model_indices for the region
                    if !haskey(unused_model_indices, region_key) ||
                       isempty(unused_model_indices[region_key])
                        unused_model_indices[region_key] = collect(1:num_models)
                    end

                    # Randomly select a model index from the unused indices, then remove it from the list
                    random_index =
                        rand(1:length(unused_model_indices[region_key]))
                    selected_model_index =
                        unused_model_indices[region_key][random_index]
                    deleteat!(unused_model_indices[region_key], random_index)
                else
                    if params.EStrategic[storage_id] == 1
                        # Sequential model selection logic
                        last_model_index_used[region_key] =
                            get(last_model_index_used, region_key, 0) + 1
                        selected_model_index =
                            (last_model_index_used[region_key] - 1) %
                            num_models + 1
                    else
                        # Control energy storage model selection logic
                        selected_model_index = 1
                    end
                end

                selected_model_key = model_keys[selected_model_index]

                push!(
                    storage_to_model_map,
                    (storage_id, models[region_key][selected_model_key]),
                )
                push!(
                    storage_to_index_map,
                    (
                        storage_id,
                        selected_model_index,
                        region_idx,
                        selected_model_key,
                    ),
                )
            end
        end
    end

    # Create DataFrame with an additional column for region
    df = DataFrame(
        StorageID = first.(storage_to_index_map),
        SelectedModelIndex = getindex.(storage_to_index_map, 2),
        Region = getindex.(storage_to_index_map, 3),
        ModelName = getindex.(storage_to_index_map, 4),
    )
    CSV.write(
        joinpath(output_folder * "/Strategic", "storage_to_index_map.csv"),
        df,
    )

    return storage_to_model_map
end

function update_battery_storage!(
    params::STESTS.ModelParams,
    ControlES::Bool,
    ratio::Float64,
    output_folder::String,
    heto::Bool,
    ESAdjustment::Float64,
    LDESEta::Vector{Float64},
)
    # Update the battery storage capacity
    for i in eachindex(params.Eeta)
        if params.Eeta[i] == 0.9 || params.Eeta[i] in LDESEta  # Adjust condition to include new efficiencies
            params.EPC[i] *= ESAdjustment
            params.EPD[i] *= ESAdjustment
            params.ESOC[i] *= ESAdjustment
            params.ESOCini[i] *= ESAdjustment
        end
    end
    if heto
        # Retrieve the new entries and their corresponding regions from params
        # Initialize temporary storage for new discrete entries
        new_storagemap = copy(params.storagemap)
        new_Eeta = Float64[]
        new_EPC = Float64[]
        new_EPD = Float64[]
        new_ESOC = Float64[]
        new_ESOCini = Float64[]
        new_EStrategic = Int64[]
        new_EMC = Float64[]
        remove_indices = Int[]

        # Process each entry for discretization if Eeta == 0.9
        for i in eachindex(params.Eeta)
            if params.Eeta[i] == 0.9 || params.Eeta[i] in LDESEta  # Adjust condition to include new efficiencies
                epd = params.EPD[i]
                eeta = params.Eeta[i]
                emc = params.EMC[i]
                append!(remove_indices, i)  # Mark this index for removal

                # Decompose EPD
                components = [1000, 100, 10, 1]
                for comp in components
                    count = floor(epd / comp)
                    epd -= count * comp
                    for _ in 1:count
                        append!(new_Eeta, eeta)
                        append!(new_EPC, comp)
                        append!(new_EPD, comp)
                        append!(new_ESOC, 4 * comp)
                        append!(new_ESOCini, 2 * comp)
                        append!(new_EStrategic, 0)  # Assuming true strategic value as integer
                        append!(new_EMC, emc)

                        # Create and append new row for storagemap corresponding to the region
                        new_storagemap =
                            vcat(new_storagemap, params.storagemap[i, :]')  # Append column-wise
                    end
                end

                # Handle residual
                if epd > 0
                    append!(new_Eeta, eeta)
                    append!(new_EPC, epd)
                    append!(new_EPD, epd)
                    append!(new_ESOC, 4 * epd)
                    append!(new_ESOCini, 2 * epd)
                    append!(new_EStrategic, 0)  # Assuming true strategic value as integer
                    append!(new_EMC, emc)

                    # Create and append new row for storagemap corresponding to the region
                    new_storagemap =
                        vcat(new_storagemap, params.storagemap[i, :]')  # Append column-wise
                end
            end
        end

        # Update params with the new discrete storages
        params.Eeta = vcat(params.Eeta, new_Eeta)
        params.EPC = vcat(params.EPC, new_EPC)
        params.EPD = vcat(params.EPD, new_EPD)
        params.ESOC = vcat(params.ESOC, new_ESOC)
        params.ESOCini = vcat(params.ESOCini, new_ESOCini)
        params.EStrategic = vcat(params.EStrategic, new_EStrategic)
        params.EMC = vcat(params.EMC, new_EMC)
        params.storagemap = new_storagemap

        # # Remove aggregated entries
        remove_indices = sort(unique(remove_indices))
        mask = trues(length(params.Eeta))
        mask[remove_indices] .= false
        params.Eeta = params.Eeta[mask]
        params.EPC = params.EPC[mask]
        params.EPD = params.EPD[mask]
        params.ESOC = params.ESOC[mask]
        params.ESOCini = params.ESOCini[mask]
        params.EStrategic = params.EStrategic[mask]
        params.EMC = params.EMC[mask]
        params.storagemap = params.storagemap[mask, :]

        # save the new configurations to CSV
        df = DataFrame(
            Eeta = new_Eeta,
            EPC = new_EPC,
            EPD = new_EPD,
            ESOC = new_ESOC,
            ESOCini = new_ESOCini,
            EStrategic = new_EStrategic,
            EMC = new_EMC,
        )
        CSV.write(joinpath(output_folder * "/Strategic", "ADDED_ES.csv"), df)
    end

    new_Eeta = []
    new_EPC = []
    new_EPD = []
    new_ESOC = []
    new_ESOCini = []
    new_EStrategic = []
    new_EMC = []

    if ratio == 1.0
        println("All AI-Powered BES.")
        for i in eachindex(params.Eeta)
            if params.Eeta[i] == 0.8
                params.EStrategic[i] = 0
            elseif params.Eeta[i] == 0.9 || params.Eeta[i] in LDESEta  # Adjust condition to include new efficiencies
                params.EStrategic[i] = 1
            else
                # Handle unexpected case, if necessary
                println(
                    "Unexpected value in params.Eeta at index $i: ",
                    params.Eeta[i],
                )
            end
        end
    elseif ratio == 0.0
        println("No AI-Powered BES.")
    else
        if heto
            error("No heterogenous ES if ISR is not 0.0 or 1.0")
        end
        println("ISR AI-Powered BES.")
        # Assume additional fields in Params for simplicity of explanation
        new_storagemap = copy(params.storagemap)

        # Track indices to remove
        remove_indices = []

        for region in 1:size(params.storagemap, 2)
            # Find battery storage indices in this region
            battery_indices = findall(
                i ->
                    (params.Eeta[i] == 0.9 || params.Eeta[i] in LDESEta) &&  # Adjust condition to include new efficiencies
                        params.storagemap[i, region] == 1,
                1:length(params.Eeta),
            )

            if isempty(battery_indices)
                continue
            end

            for battery_index in battery_indices
                epc = params.EPC[battery_index]
                epd = params.EPD[battery_index]
                eeta = params.Eeta[battery_index]
                esoc = params.ESOC[battery_index]
                esocini = params.ESOCini[battery_index]
                emc = params.EMC[battery_index]

                append!(new_Eeta, [eeta, eeta])
                append!(new_EPC, [epc * ratio, epc * (1 - ratio)])
                append!(new_EPD, [epd * ratio, epd * (1 - ratio)])
                append!(new_ESOC, [esoc * ratio, esoc * (1 - ratio)])
                append!(new_ESOCini, [esocini * ratio, esocini * (1 - ratio)])
                append!(new_EStrategic, [1, 0])
                append!(new_EMC, [emc, emc])
                # # Aggregate values
                # total_EPC = sum(params.EPC[battery_indices])
                # total_EPD = sum(params.EPD[battery_indices])
                # total_ESOC = sum(params.ESOC[battery_indices])
                # total_ESOCini = sum(params.ESOCini[battery_indices])

                # Prepare data for two new storages
                # append!(new_Eeta, [0.9, 0.9])
                # append!(new_EPC, [total_EPC * ratio, total_EPC * (1 - ratio)])
                # append!(new_EPD, [total_EPD * ratio, total_EPD * (1 - ratio)])
                # append!(new_ESOC, [total_ESOC * ratio, total_ESOC * (1 - ratio)])
                # append!(
                #     new_ESOCini,
                #     [total_ESOCini * ratio, total_ESOCini * (1 - ratio)],
                # )
                # append!(new_EStrategic, [1, 0])

                # Update storagemap for new entries
                # Assuming new_row_strategic is a 1xN vector
                new_row_strategic = zeros(Int64, size(new_storagemap, 2))
                new_row_strategic[region] = 1

                # To add two identical rows to the matrix, ensure they are treated as separate rows
                new_storagemap = vcat(
                    new_storagemap,
                    reshape(new_row_strategic, 1, :),
                    reshape(new_row_strategic, 1, :),
                )

                append!(remove_indices, battery_indices)
            end
        end

        # Now, remove the aggregated entries and update params
        # Append new entries
        params.Eeta = vcat(params.Eeta, new_Eeta)
        params.EPC = vcat(params.EPC, new_EPC)
        params.EPD = vcat(params.EPD, new_EPD)
        params.ESOC = vcat(params.ESOC, new_ESOC)
        params.ESOCini = vcat(params.ESOCini, new_ESOCini)
        params.EStrategic = vcat(params.EStrategic, new_EStrategic)
        params.EMC = vcat(params.EMC, new_EMC)

        # Create a mask that is true for indices that should be kept
        remove_indices = sort(unique(remove_indices))
        mask = trues(length(params.Eeta))
        mask[remove_indices] .= false
        params.Eeta = params.Eeta[mask]
        params.EPC = params.EPC[mask]
        params.EPD = params.EPD[mask]
        params.ESOC = params.ESOC[mask]
        params.ESOCini = params.ESOCini[mask]
        params.EStrategic = params.EStrategic[mask]
        params.EMC = params.EMC[mask]
        params.storagemap = new_storagemap[mask, :]
    end

    #Hard Code Here!
    if ControlES
        Control_Eeta = 0.9
        Control_EPC = 1.0
        Control_EPD = 1.0
        Control_ESOC = 4.0
        Control_ESOCini = 2.0
        Control_EStrategic = 2
        Control_EMC = 20.0
        Control_Storagemap = [0, 1, 0, 0, 0, 0]'

        # Append control storage parameters to the existing arrays in `params`
        push!(params.Eeta, Control_Eeta)
        push!(params.EPC, Control_EPC)
        push!(params.EPD, Control_EPD)
        push!(params.ESOC, Control_ESOC)
        push!(params.ESOCini, Control_ESOCini)
        push!(params.EStrategic, Control_EStrategic)
        push!(params.EMC, Control_EMC)
        params.storagemap = vcat(params.storagemap, Control_Storagemap)

        append!(new_Eeta, Control_Eeta)
        append!(new_EPC, Control_EPC)
        append!(new_EPD, Control_EPD)
        append!(new_ESOC, Control_ESOC)
        append!(new_ESOCini, Control_ESOCini)
        append!(new_EStrategic, Control_EStrategic)
        append!(new_EMC, Control_EMC)
    end

    # Save the new entries to csv
    df = DataFrame(
        Eeta = new_Eeta,
        EPC = new_EPC,
        EPD = new_EPD,
        ESOC = new_ESOC,
        ESOCini = new_ESOCini,
        EStrategic = new_EStrategic,  # Assuming you also have this array
        EMC = new_EMC,
    )

    CSV.write(joinpath(output_folder * "/Strategic", "ADDED_ES.csv"), df)

    return params
end

function CalcValueNoUnc(d, c, eta, vi, iC, iD)
    """
    ####### Completely translated to Julia, gave exact numbers to Python #######

    Title: Calculate Risk-Neutral value function using deterministic price
    Inputs:
        d - price right now
        c - marginal discharge cost
        eta - efficiency
        vi - input value function for the next time period, which equals to
        v_t(e) where e is sampled from 0 to 1 at the granularity e
    Outputs:
        vo - value function for the current time period sampled at ed
    """
    # add a large number of upper and lower v, where the first point is
    # v_t(0-) = +infty, and the second point is v_t(0), the second largest one is
    # v_t(1), and the largest one is v_t(1+) = -infty
    lNum = fill(1e5)
    v_foo = [lNum; vi; -lNum]

    # calculate soc after charge vC = v_t(e+P*eta)
    vC = v_foo[iC]

    # calculate soc after discharge vC = v_t(e-P/eta)
    vD = v_foo[iD]

    # calculate CDF and PDF
    FtEC = (vi .* eta .> d)
    FtCC = (vC .* eta .> d)
    FtED = ((vi ./ eta .+ c) .* ((vi ./ eta .+ c) .> 0) .> d)
    FtDD = ((vD ./ eta .+ c) .* ((vD ./ eta .+ c) .> 0) .> d)

    # calculate terms
    Term1 = vC .* FtCC
    Term2 = d .* (vC .* eta .<= d) .* (vi .* eta .> d) ./ eta
    Term3 = vi .* (FtED - FtEC)
    Term4 =
        d .* (((vi ./ eta .+ c) .* ((vi ./ eta .+ c) .> 0)) .<= d) .*
        (((vD ./ eta .+ c) .* ((vD ./ eta .+ c) .> 0)) .> d) .* eta
    Term5 = -c .* eta .* (FtDD - FtED)
    Term6 = vD .* (1 .- FtDD)

    # output new value function sampled at ed
    vo = Term1 + Term2 + Term3 + Term4 + Term5 + Term6
    return vo
end

function add_long_duration_storage!(
    params::STESTS.ModelParams,
    LDESRatio::Vector{Float64},
    LDESDur::Vector{Int},
    LDESEta::Vector{Float64},
    LDESMC::Vector{Float64},
    output_folder::String,
)
    """
    ### Function used to add new LDES base on current battery energy storage capacity###
    Inputs:
        params - system data
        LDESRatio - New energy storage capacity proportional to the original energy storage capacity
        LDESDur - Duration of new energy storage
        LDESEta - One-way efficiency of new energy storage
        LDESMC - Marginal cost of new energy storage
        output_folder - output folder
    Outputs: 
        params - Updated system data with new energy storage
        saved csv file with new energy storage data - ADDED_LDES.csv
    """
    new_Eeta = Float64[]
    new_EPC = Float64[]
    new_EPD = Float64[]
    new_ESOC = Float64[]
    new_ESOCini = Float64[]
    new_EStrategic = Int64[]
    new_EMC = Float64[]
    new_storagemap = copy(params.storagemap)

    for i in eachindex(params.Eeta)
        if params.Eeta[i] == 0.9
            original_EPC = params.EPC[i]
            original_EPD = params.EPD[i]
            original_strategic = params.EStrategic[i]
            original_storagemap = params.storagemap[i, :]

            for j in 1:length(LDESRatio)
                if LDESRatio[j] == 0
                    continue  # Skip if the ratio is 0
                end
                ratio = LDESRatio[j]
                duration = LDESDur[j]
                eta = LDESEta[j]
                emc = LDESMC[j]
                new_EPC_val = original_EPC * ratio
                new_EPD_val = original_EPD * ratio
                new_ESOC_val = duration * new_EPC_val
                new_ESOCini_val = 0.5 * new_ESOC_val

                append!(new_Eeta, eta)
                append!(new_EPC, new_EPC_val)
                append!(new_EPD, new_EPD_val)
                append!(new_ESOC, new_ESOC_val)
                append!(new_ESOCini, new_ESOCini_val)
                append!(new_EStrategic, original_strategic)
                append!(new_EMC, emc)

                new_storagemap =
                    vcat(new_storagemap, reshape(original_storagemap, 1, :))
            end
        end
    end

    # Update params with the new discrete storages
    params.Eeta = vcat(params.Eeta, new_Eeta)
    params.EPC = vcat(params.EPC, new_EPC)
    params.EPD = vcat(params.EPD, new_EPD)
    params.ESOC = vcat(params.ESOC, new_ESOC)
    params.ESOCini = vcat(params.ESOCini, new_ESOCini)
    params.EStrategic = vcat(params.EStrategic, new_EStrategic)
    params.EMC = vcat(params.EMC, new_EMC)
    params.storagemap = new_storagemap

    # Save the new entries to CSV
    df = DataFrame(
        Eeta = params.Eeta,
        EPC = params.EPC,
        EPD = params.EPD,
        ESOC = params.ESOC,
        ESOCini = params.ESOCini,
        EStrategic = params.EStrategic,
        EMC = params.EMC,
    )

    CSV.write(joinpath(output_folder * "/Strategic", "ADDED_LDES.csv"), df)
    return params
end

# function update_long_duration_storage!(
#     params::STESTS.ModelParams,
#     LDESRatio::Vector{Float64},
#     LDESDur::Vector{Int},
#     output_folder::String,
# )
# ```
# ### Function used to split total energy storage into multiple smaller energy storage units with different durations ###
# Inputs:
#     params - system data
#     LDESRatio - Split ratio for different duration energy storage
#     LDESDur - Duration of splited energy storage
#     output_folder - output folder
# Outputs: 
#     params - Updated system data
#     saved csv file with new energy storage data - UPDATED_LDES.csv
# ```
#     new_Eeta = Float64[]
#     new_EPC = Float64[]
#     new_EPD = Float64[]
#     new_ESOC = Float64[]
#     new_ESOCini = Float64[]
#     new_EStrategic = Int64[]
#     new_storagemap = copy(params.storagemap)
#     remove_indices = Int[]

#     for i in eachindex(params.Eeta)
#         if params.Eeta[i] == 0.9
#             original_EPC = params.EPC[i]
#             original_EPD = params.EPD[i]
#             original_ESOC = params.ESOC[i]
#             original_ESOCini = params.ESOCini[i]
#             original_Eta = params.Eeta[i]
#             original_strategic = params.EStrategic[i]
#             original_storagemap = params.storagemap[i, :]

#             for j in 1:length(LDESRatio)
#                 ratio = LDESRatio[j]
#                 duration = LDESDur[j]
#                 new_EPC_val = original_EPC * ratio
#                 new_EPD_val = original_EPD * ratio
#                 new_ESOC_val = duration * new_EPC_val
#                 new_ESOCini_val = 0.5 * new_ESOC_val

#                 append!(new_Eeta, original_Eta)
#                 append!(new_EPC, new_EPC_val)
#                 append!(new_EPD, new_EPD_val)
#                 append!(new_ESOC, new_ESOC_val)
#                 append!(new_ESOCini, new_ESOCini_val)
#                 append!(new_EStrategic, original_strategic)

#                 new_storagemap =
#                     vcat(new_storagemap, reshape(original_storagemap, 1, :))
#             end

#             append!(remove_indices, i)
#         end
#     end

#     params.Eeta = vcat(params.Eeta, new_Eeta)
#     params.EPC = vcat(params.EPC, new_EPC)
#     params.EPD = vcat(params.EPD, new_EPD)
#     params.ESOC = vcat(params.ESOC, new_ESOC)
#     params.ESOCini = vcat(params.ESOCini, new_ESOCini)
#     params.EStrategic = vcat(params.EStrategic, new_EStrategic)
#     params.storagemap = new_storagemap

#     # Remove original entries that were split
#     remove_indices = sort(unique(remove_indices))
#     mask = trues(length(params.Eeta))
#     mask[remove_indices] .= false
#     params.Eeta = params.Eeta[mask]
#     params.EPC = params.EPC[mask]
#     params.EPD = params.EPD[mask]
#     params.ESOC = params.ESOC[mask]
#     params.ESOCini = params.ESOCini[mask]
#     params.EStrategic = params.EStrategic[mask]
#     params.storagemap = params.storagemap[mask, :]

#     # Save the new entries to CSV
#     df = DataFrame(
#         Eeta = params.Eeta,
#         EPC = params.EPC,
#         EPD = params.EPD,
#         ESOC = params.ESOC,
#         ESOCini = params.ESOCini,
#         EStrategic = params.EStrategic,
#     )

#     CSV.write(joinpath(output_folder * "/Strategic", "UPDATED_LDES.csv"), df)
#     return params
# end

function generate_value_function(
    T,
    P,
    eta,
    num_segment,
    RTP;
    c = 20.0,
    ed = 0.001,
    Ne = 1001,
    ef = 0.5,
)
    """
    ####### Completely translated to Julia, gave exact numbers to Python #######
    Generate value function v and downsampled value function vAvg
    """

    # start_time = time()

    # Set final SoC level
    vEnd = zeros(Ne)
    vEnd[1:floor(Int64, ef * Ne)] .= 1e2 # Use 100 as the penalty for final discharge level

    # Define the risk-neutral value function and populate the final column.
    # v[1, 1] is the marginal value of 0% SoC at the beginning of day 1, v[Ne, T] is the marginal value of 100% SoC at the beginning of the last operating day
    v = zeros(Ne, T + 1) # initialize the value function series
    v[:, end] = vEnd  # v.shape == (1001, 210241)

    # Process indices: discretize vt by modeling it as a vector v_{t,j} in which each element is associated with equally spaced SoC samples
    es = collect(0:ed:1)

    # the number of samples is J = 1 + E/ed
    Ne = length(es)

    # Calculate SoC after charge vC = v_t(e+P*eta)
    eC = es .+ P * eta  # [0.0375, 0.0385, 0.0395, ..., 1.0355, 1.0365, 1.0375]
    iC = max.(0, min.(Ne + 1, ceil.(Int, eC / ed))) .+ 1

    # Calculate SoC after discharge vC = v_t(e-P/eta)
    eD = es .- P / eta
    iD = max.(0, min.(Ne + 1, floor.(Int, eD / ed))) .+ 1

    # Populate value function
    for t in T:-1:1 # start from the last day and move backwards
        vi = v[:, t+1] # input value function of next time stamp
        vo = CalcValueNoUnc(RTP[t], c, eta, vi, iC, iD)
        v[:, t] = vo # record the result
    end

    # end_time = time()
    # println("Time:", end_time - start_time)

    # Downsample
    vAvg = reshape(
        reduce(
            vcat,
            [
                mean(
                    v[
                        (i-1)*Int(1 / ed / num_segment)+1:i*Int(
                            1 / ed / num_segment,
                        ),
                        :,
                    ],
                    dims = 1,
                ) for i in 1:num_segment
            ],
        ),
        num_segment,
        :,
    )

    return vAvg
end