

using MinimalRLCore

mutable struct Critterbot <: AbstractEnvironment
    num_steps::Int
    num_features::Int
    num_targets::Int
    sensors::Vector{Int}

    idx::Int
    data::Array{Float64}
end

function Critterbot(obs_sensors, target_sensors)
    all_sensors = vcat(obs_sensors, target_sensors)
    num_features = length(obs_sensors)
    return Critterbot(CritterbotUtils.numSteps(),
                      num_features, length(target_sensors), CritterbotUtils.getSensorIndices(all_sensors),
                      0, squish(CritterbotUtils.loadSensor(all_sensors)))
end

function Critterbot(obs_sensors, target_sensors, γs::AbstractArray)
    all_sensors = vcat(obs_sensors, target_sensors)
    num_features = length(obs_sensors)*length(γs)
    data = squish(vcat(CritterbotUtils.getReturns(obs_sensors, γs), CritterbotUtils.loadSensor(target_sensors)))
    
    return Critterbot(CritterbotUtils.numSteps(),
                      num_features, length(target_sensors), CritterbotUtils.getSensorIndices(all_sensors),
                      0, data)
end

Critterbot(obs_sensors, target_sensors, γ_str::String) =
    Critterbot(obs_sensors, target_sensors, eval(Meta.parse(γ_str)))

# Hack to use same features as targets; just duplicate the data in new cols
Critterbot(sensors::Vector{Int}) = Critterbot(sensors, sensors)
get_num_features(cb::Critterbot) = cb.num_features
get_num_targets(cb::Critterbot) = cb.num_targets

function MinimalRLCore.start!(cb::Critterbot)
    cb.idx = 1
    return MinimalRLCore.get_state(cb)
end

function MinimalRLCore.step!(cb::Critterbot)
    cb.idx += 1
    return MinimalRLCore.get_state(cb), MinimalRLCore.get_reward(cb)
end

# Data for each sensor in a row, so that we can access data for all sensors by col
MinimalRLCore.get_state(cb::Critterbot) = cb.data[1:cb.num_features, cb.idx]
MinimalRLCore.get_reward(cb::Critterbot) = cb.data[cb.num_features+1:end, cb.idx] #


mutable struct CritterbotTPC <: AbstractEnvironment
    num_steps::Int
    num_features::Int

    idx::Int
    obs_data::Array{Float64}
    rewards::Array{Float64}
    discounts::Array{Float64}
    the_all_seeing_eye::Array{Float64}
end

function CritterbotTPC(obs_sensors; γ=0.9875)
    # all_sensors = vcat(obs_sensors, target_sensors)

    volts = CritterbotUtils.loadSensor(["Motor$(i)0" for i in 0:2])
    cur = CritterbotUtils.loadSensor(["Motor$(i)2" for i in 0:2])

    rewards = sum(abs.(cur .* volts); dims=1)
    light3 = GVFN.CritterbotUtils.loadSensor("Light3")
    discounts = (light3 .< 1020) .* γ

    num_features = length(obs_sensors)
    feats = squish(CritterbotUtils.loadSensor(obs_sensors))

    the_all_seeing_eye = CritterbotUtils.loadSensor("powerToGoal-$(γ)")
    
    return CritterbotTPC(
        CritterbotUtils.numSteps(),
        num_features,        
        0,
        feats,
        rewards,
        discounts,
        the_all_seeing_eye)
end

function CritterbotTPC(obs_sensors, γs::AbstractArray; γ=0.9875)
    # all_sensors = vcat(obs_sensors, target_sensors)
    num_features = length(obs_sensors)*length(γs)
    feats = squish(CritterbotUtils.getReturns(obs_sensors, γs))

    volts = CritterbotUtils.loadSensor(["Motor$(i)0" for i in 0:2])
    cur = CritterbotUtils.loadSensor(["Motor$(i)2" for i in 0:2])

    rewards = sum(abs.(cur .* volts); dims=1)
    light3 = GVFN.CritterbotUtils.loadSensor("Light3")
    discounts = (light3 .< 1020).*γ

    the_all_seeing_eye = CritterbotUtils.loadSensor("powerToGoal-$(γ)")
    
    return CritterbotTPC(
        CritterbotUtils.numSteps(),
        num_features,        
        0,
        feats,
        rewards,
        discounts,
        the_all_seeing_eye)
end

CritterbotTPC(obs_sensors, γ_str::String) =
    CritterbotTPC(obs_sensors, eval(Meta.parse(γ_str)))

# Hack to use same features as targets; just duplicate the data in new cols
# CritterbotTPC(sensors::Vector{Int}) = Critterbot(sensors, sensors)
get_num_features(cb::CritterbotTPC) = cb.num_features
# get_num_targets(cb::CritterbotTPC) = 1

function MinimalRLCore.start!(cb::CritterbotTPC)
    cb.idx = 1
    return MinimalRLCore.get_state(cb)
end

function MinimalRLCore.step!(cb::CritterbotTPC)
    cb.idx += 1
    return MinimalRLCore.get_state(cb), MinimalRLCore.get_reward(cb)
end

# Data for each sensor in a row, so that we can access data for all sensors by col
MinimalRLCore.get_state(cb::CritterbotTPC) = cb.obs_data[:, cb.idx]
MinimalRLCore.get_reward(cb::CritterbotTPC) = [cb.rewards[cb.idx], cb.discounts[cb.idx]]
ground_truth(cb::CritterbotTPC) = [cb.the_all_seeing_eye[cb.idx]]

