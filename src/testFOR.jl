# using STESTS, JuMP, Gurobi, CSV, DataFrames, Statistics, SMTPClient, Random

# Year = 2022
# # Read data from .jld2 file 
# params = STESTS.read_jld2(
#     "./data/ADS2032_5GWBES_BS_AggES_" * "$Year" * "_fixed.jld2",
# )

# function generate_real_time_outage_profiles(
#     GFOR::Vector{Float64},
#     GOD::Vector{Int64},
#     NDay::Int,
#     seed::Int,
# )
#     Random.seed!(seed)  # Set the random seed for reproducibility
#     num_generators = length(GFOR)
#     hours_per_year = 8760
#     outage_profile = ones(Int, num_generators, hours_per_year)

#     # Generate real-time outage profile
#     for i in 1:num_generators
#         total_outage_hours = round(Int, hours_per_year * GFOR[i])
#         outage_duration = GOD[i]
#         remaining_outage_hours = total_outage_hours

#         while remaining_outage_hours > 0
#             if remaining_outage_hours >= outage_duration
#                 outage_start = rand(1:(hours_per_year-outage_duration+1))
#                 outage_end =
#                     min(outage_start + outage_duration - 1, hours_per_year)
#                 actual_outage_duration = outage_end - outage_start + 1

#                 # Ensure no overlap of outages
#                 if sum(outage_profile[i, outage_start:outage_end]) ==
#                    actual_outage_duration
#                     outage_profile[i, outage_start:outage_end] .= 0
#                     remaining_outage_hours -= actual_outage_duration
#                 end
#             end

#             # If remaining_outage_hours is less than outage_duration, we need to handle it separately
#             if remaining_outage_hours > 0 &&
#                remaining_outage_hours < outage_duration
#                 outage_start = rand(1:(hours_per_year-remaining_outage_hours+1))
#                 outage_end = outage_start + remaining_outage_hours - 1

#                 if sum(outage_profile[i, outage_start:outage_end]) ==
#                    remaining_outage_hours
#                     outage_profile[i, outage_start:outage_end] .= 0
#                     remaining_outage_hours = 0
#                 end
#             end
#         end
#     end
#     return outage_profile
# end

# function generate_day_ahead_profile(outage_profile)
#     n_generators, n_hours = size(outage_profile)
#     hours_per_day = 24
#     day_ahead_profile = copy(outage_profile)  # Start with a copy of the real-time profile

#     for g in 1:n_generators
#         is_outage = false
#         start_hour = 0

#         for h in 1:n_hours
#             if outage_profile[g, h] == 0
#                 if !is_outage
#                     is_outage = true
#                     start_hour = h
#                     # Calculate the day containing the first zero and set it to all ones in the day-ahead profile
#                     start_day = div(start_hour - 1, hours_per_day)
#                     day_start_hour = start_day * hours_per_day + 1
#                     day_end_hour = (start_day + 1) * hours_per_day
#                     day_ahead_profile[
#                         g,
#                         day_start_hour:min(day_end_hour, n_hours),
#                     ] .= 1
#                 end
#             else
#                 if is_outage
#                     is_outage = false
#                 end
#             end
#         end
#     end

#     return day_ahead_profile
# end

# # Example usage:
# NDay = 364  # Number of days
# seed = 123  # Random seed

# # Generate and save the outage profiles
# outage_profile =
#     generate_real_time_outage_profiles(params.GFOR, params.GOD, NDay, seed)
# RTO_repeated = repeat(outage_profile, inner = (1, 12))

# # Generate the day-ahead outage profile
# day_ahead_profile = generate_day_ahead_profile(outage_profile)

# # Convert the matrices to DataFrames
# df_outage_profile = DataFrame(outage_profile, :auto)

# # Save the DataFrames to CSV files
# CSV.write("outage_profile.csv", df_outage_profile)

# # Convert the matrix to a DataFrame
# df_day_ahead_profile = DataFrame(day_ahead_profile, :auto)

# # Save the DataFrame to a CSV file
# CSV.write("day_ahead_outage_profile.csv", df_day_ahead_profile)

RTOInput = [
    0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
]

# Define the V matrix
V = [
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
]

# Modify V based on the first 1 after a 0 in RTOInput
for i in axes(V, 1)
    zero_indices = findall(x -> x == 0, RTOInput[i, :])

    for index in zero_indices
        if index < size(RTOInput, 2) && RTOInput[i, index+1] == 1
            V[i, index+1] = 1
        end
    end
end

# Print the modified V
println("Modified V:")
println(V)