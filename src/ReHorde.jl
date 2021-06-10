module ReHorde

import LinearAlgebra: dot

using LoopVectorization

using Tullio

struct FeatureCumulant
    idx::Int
end
(c::FeatureCumulant)(o, x::Vector{Int}) = c.idx ∈ x

struct ObservationCumulant
    idx::Int
end
(c::ObservationCumulant)(o, x) = o[c.idx]

struct OnPolicy end

# What are we doing.
# Massively parallel GVF learning
struct GVF{W, Z, C, Π, Γ}
    w::W
    z::Z
    c::C
    π::Π
    γ::Γ
end

(gvf::GVF{<:AbstractVector})(x) = dot(gvf.w, x)
(gvf::GVF{<:AbstractVector})(x::AbstractVector{Int}) = begin; w = gvf.w; @tullio ret := w[x[i]]; end #sum(@view gvf.w[x])

struct TDλ
    α::Float32
    λ::Float32
end

function update!(lu::TDλ, gvf::GVF, o_t, a_t, o_tp1, x_t, x_tp1, r, μ_t)

    z = gvf.z
    w = gvf.w
    
    λ = lu.λ
    ρ_t = if gvf.π isa OnPolicy
        1f0
    else
        gvf.π(o_t, a_t)/μ_t
    end
    γ_t = if gvf.γ isa AbstractFloat
        gvf.γ
    else
        gvf.γ(o_t)
    end
    c = gvf.c(o_tp1, x_tp1)
    δ = c + γ_t*gvf(x_tp1) - gvf(x_t)
    
    if eltype(x_t) <: Integer
        z .*= γ_t*λ
        # view(z, x_t) .+= 1
        @tullio z[x_t[i]] = z[x_t[i]] + 1
        z .*= ρ_t
        w .+= (lu.α * δ) .* z
    else
        gvf.z .= ρ_t .* ((γ_t*λ) * gvf.z .+ x_t)
        gvf.w .+= lu.α * δ * gvf.z
    end
end


function update!(lu, gvfs::Vector{G}, o_t, a_t, o_tp1, x_t, x_tp1, r, μ_t) where G<:GVF
    Threads.@threads for i in 1:length(gvfs)
	update!(lu, gvfs[i], o_t, a_t, o_tp1, x_t, x_tp1, r, μ_t)
    end
end


# Write your package code here.
include("critterbot_utils.jl")
include("Critterbot.jl")


import ProgressMeter: @showprogress
const CBU = ReHorde.CritterbotUtils

function main_experiment()
    
    tiled_features_raw = CBU.loadTiles()
    tiled_features = [tiled_features_raw[i, :] for i in 1:size(tiled_features_raw, 1)]
    sensors_raw = Float32.(CBU.relevant_sensors())
    sensors = [sensors_raw[i, :] for i in 1:size(sensors_raw, 1)]

    horde = vcat([[
        [GVF(
	    zeros(Float32, CBU.numFeatures()),
	    zeros(Float32, CBU.numFeatures()),
	    ReHorde.ObservationCumulant(i),
	    ReHorde.OnPolicy(),
	    γ) for i in 1:length(sensors[1])];
        [GVF(
	    zeros(Float32, CBU.numFeatures()),
	    zeros(Float32, CBU.numFeatures()),
	    ReHorde.FeatureCumulant(i),
	    ReHorde.OnPolicy(),
	    γ) for i in rand(1:CBU.numFeatures(), 487)]]
                  for γ in [0.0, 0.8, 0.95, 0.9875]]...)

    @show length(horde)

    lu = TDλ(0.1/length(tiled_features[1]), 0.9)

    @showprogress 0.1 "Time Step: " for t = 2:length(tiled_features)
        x_t = tiled_features[t-1]
        x_tp1 = tiled_features[t]
        o_t = sensors[t-1]
        o_tp1 = sensors[t]
        a = nothing
        μ = nothing
        r = nothing
        
        update!(lu, horde, o_t, a, o_tp1, x_t, x_tp1, r, μ)
    end

    
    
end


end
