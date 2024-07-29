__precompile__
mutable struct Settings
    # grid settings
    # number spatial interfaces
    Nx::Int64;
    # number spatial cells
    NCells::Int64;
    # number of collocation points
    Nxi::Int64; 
    Neta::Int64; 
    # start and end point
    a::Float64;
    b::Float64;
    # grid cell width
    Δx::Float64

    # time settings
    # end time
    tEnd::Float64;
    # time increment
    Δt::Float64;
    # CFL number 
    cfl::Float64;
    
    # degree PN
    nPN::Int;

    # spatial grid
    x
    xMid

    # physical parameters
    σₐ::Float64;
    σₛ::Float64;
    σₛξ::Float64;
    σₛη::Float64;

    # truncation tolerance
    ϵ::Float64

    # low rank parameters
    r::Int;
    rMax::Int;
    rMin::Int;

    problem::String

    function Settings(Nx::Int=202,nPN::Int=100,Nxi::Int=100,Neta::Int=100,problem="radiationUQ8D")
        # spatial grid setting
        NCells = Nx - 1;
        a = -2.5; # left boundary
        b = 2.5; # right boundary
        
        # time settings
        tEnd = 2#0.75;
        cfl = 0.1; # CFL condition

        x = collect(range(a,stop = b,length = NCells));
        Δx = x[2]-x[1];
        x = [x[1]-Δx;x]; # add ghost cells so that boundary cell centers lie on a and b
        x = x.+Δx/2;
        xMid = x[1:(end-1)].+0.5*Δx

        # physical parameters
        sigmaS = 5.0#5.0; # σₛ = sigmaS(x) + ξ ⋅  σₛξ + η ⋅ σₛη
        sigmaA = 0.0;   
        σₛξ = 4.0#4.0
        σₛη = 1.0#1.0

        r = 10;
        ϵ = 1e-3;

        dt = cfl*Δx;

        rMax = 50;
        rMin = 2;

        # build class
        new(Nx,NCells,Nxi,Neta,a,b,Δx,tEnd,dt,cfl,nPN,x,xMid,sigmaA,sigmaS,σₛξ,σₛη,ϵ,r,rMax,rMin,problem);
    end

end

function IC(x,xi=0.0)
    y = zeros(size(x));
    σ² = 0.03^2
    x0 = 0.0
    for (j, xⱼ) in enumerate(x);
        y[j] = 1/(sqrt(2*pi*σ²)) * exp(-((xⱼ-x0)*(xⱼ-x0))/2.0/σ²)
    end
    return y;
end

function refSol(obj::Settings)
    id = Diagonal(ones(2))
    σx = [0.0 1.0; 1.0 0.0]
    σz = [1.0 0.0; 0.0 -1.0]
    Ω = 1
    Dₓₓ = Ω * σx

    id = Diagonal(ones(2))
    σx = [0.0 1.0; 1.0 0.0]
    σz = [1.0 0.0; 0.0 -1.0]
    Ω = 1
    Dₓₓ = Ω * σx

    RhsTerm = [Dₓₓ, id, id, id]
    push!(ARhs,RhsTerm)
    RhsTerm = [id, Dₓₓ, id, id]
    push!(ARhs,RhsTerm)
    RhsTerm = [id, id, Dₓₓ, id]
    push!(ARhs,RhsTerm)
    RhsTerm = [id, id, id, Dₓₓ]
    push!(ARhs,RhsTerm)
    RhsTerm = [σz, σz, id, id]
    push!(ARhs,RhsTerm)
    RhsTerm = [id, σz, σz, id]
    push!(ARhs,RhsTerm)
    RhsTerm = [id, id, σz, σz]
    push!(ARhs,RhsTerm)
    RhsTerm = [σz, id, id, σz]

    M = zeros(2^4, 2^4)
    for rhs in ARhs
        tmp = ones(1)
        for A in ARhs
            tmp = kron(tmp, A)
        end
        M += tmp
    end

    # exp( -1im * M * obj.settings.tEnd )
    return exp(M * obj.settings.tEnd)

end