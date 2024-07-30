__precompile__

using LinearAlgebra
using LegendrePolynomials
using QuadGK
using TensorToolbox
using PyCall
np = pyimport("numpy")

struct SolverDeterministic
    # Solver settings
    settings::Settings;
    
    # squared L2 norms of Legendre coeffs
    γ::Array{Float64,1};
    # flux matrix PN system
    A::Array{Float64,2};
    # Roe matrix
    AbsA::Array{Float64,2};

    # stencil matrices
    Dₓ::Tridiagonal{Float64, Vector{Float64}};
    Dₓₓ::Tridiagonal{Float64, Vector{Float64}};

    # physical parameters
    σₐ::Diagonal{Float64, Vector{Float64}};
    σₛ::Diagonal{Float64, Vector{Float64}};
    σₛξ::Diagonal{Float64, Vector{Float64}};
    σₛη::Diagonal{Float64, Vector{Float64}};

    wξ::Vector{Float64}
    wη::Vector{Float64}

    G::Diagonal{Float64, Vector{Float64}};

    # constructor
    function SolverDeterministic(settings)
        nx = settings.NCells;
        nξ = settings.Nxi;
        nη = settings.Neta;
        nΩ = settings.nPN
        Δx = settings.Δx;

        # setup flux matrix
        γ = ones(nΩ);

        # setup γ vector
        γ = zeros(nΩ);
        for i = 1:nΩ
            n = i-1;
            γ[i] = 2/(2*n+1);
        end
        
        # setup flux matrix
        A = zeros(nΩ,nΩ)

        for i = 1:(nΩ-1)
            n = i-1;
            A[i,i+1] = (n+1)/(2*n+1)*sqrt(γ[i+1])/sqrt(γ[i]);
        end

        for i = 2:nΩ
            n = i-1;
            A[i,i-1] = n/(2*n+1)*sqrt(γ[i-1])/sqrt(γ[i]);
        end

        # setup Roe matrix
        S = eigvals(A)
        V = eigvecs(A)
        AbsA = V*abs.(Diagonal(S))*inv(V)

        # set up spatial stencil matrices
        Dₓ = Tridiagonal(-ones(nx-1)./Δx/2.0,zeros(nx),ones(nx-1)./Δx/2.0) # central difference matrix
        Dₓₓ = Tridiagonal(ones(nx-1)./Δx/2.0,-ones(nx)./Δx,ones(nx-1)./Δx/2.0) # stabilization matrix

        #Compute diagonal of scattering matrix G
        G = Diagonal([0.0;ones(nΩ-1)]);
        σₛ = Diagonal(ones(nx)).*settings.σₛ;
        σₐ = Diagonal(ones(nx)).*settings.σₐ;

        ξ, wξ = gausslegendre(nξ);
        η, wη = gausslegendre(nη);
        σₛξ = Diagonal(settings.σₛξ .* ξ);
        σₛη = Diagonal(settings.σₛη .* η);

        new(settings,γ,A,AbsA,Dₓ,Dₓₓ,σₐ,σₛ,σₛξ,σₛη,wξ,wη,G);
    end
end

function SetupIC(obj::SolverDeterministic)
    u = zeros(obj.settings.NCells,obj.settings.nPN); # Nx interfaces, means we have Nx - 1 spatial cells
    u[:,1] = IC(obj.settings.xMid);
    return u;
end

function Solve(obj::SolverDeterministic,ξ::Float64=0.0,η::Float64=0.0)
    t = 0.0;
    Δt = obj.settings.Δt;
    tEnd = obj.settings.tEnd;

    nt = Int(ceil(tEnd/Δt));     # number of time steps
    Δt = obj.settings.tEnd/nt;           # adjust Δt

    N = obj.settings.nPN;
    nx = obj.settings.NCells;

    # Set up initial condition
    u = SetupIC(obj);

    #Compute diagonal of scattering matrix G
    G = Diagonal([0.0;ones(N-1)]);
    σₛ = Diagonal(ones(nx)).*obj.settings.σₛ .+ Diagonal(ones(nx)).* obj.settings.σₛξ .* ξ .+ Diagonal(ones(nx)).* obj.settings.σₛη .* η;
    σₐ = Diagonal(ones(nx)).*obj.settings.σₐ;
    A = obj.A;
    AbsA = obj.AbsA;

    F = u -> - obj.Dₓ*u*A' .+ obj.Dₓₓ*u*AbsA' .- σₐ*u .- σₛ*u*G; 

    #prog = Progress(nt,1)
    #loop over time
    for n=1:nt
        u = rk(F, u, Δt, 4)
        #u = u .- Δt * obj.Dₓ*u*A' .+ Δt * obj.Dₓₓ*u*AbsA' .- Δt * σₐ*u .- Δt * σₛ*u*G; 
        #next!(prog) # update progress bar
    end
    # return end time and solution
    return u;

end