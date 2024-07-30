clear all; close all; clc
addpath('C:\Users\Dominik\Documents\MATLAB\Low rank approximations\unconventional_TTN_integrator')
addpath('C:\Users\Dominik\Documents\MATLAB\Low rank approximations\parallel_TTN_integrator')
addpath('C:\Users\Dominik\Documents\MATLAB\Low rank approximations\rank_adaptive_integrator_for_TTN')
addpath('C:\Users\Dominik\Documents\MATLAB\Matlab toolboxes\hm-toolbox-master')
addpath('C:\Users\Dominik\Documents\MATLAB\Low rank approximations\TTNO\HSS_and_no_structure_case')

% number particles
d = 4;
l = log(d)/log(2); % number of layers
n = 2;             % physical dimension

% parameters of the model
nu = 2;
Delta = 1;
Omega = 1;
alpha = 1;
% time step size and final time
T_end = 1;

num_runs = 8;

% rank of initial data at bottom layer
r = 2;
% bond dimension operator
r_op_max = 20;
r_op_min = 2;

% for rank-adaptive integrator
tol = 10^-8;
r_min = 2;
r_max = 30;

% constant for step rejection
const = 10;

% initialisations
sx=[0,1;1,0];  %% Pauli Matrix x
sy=[0,-1i;1i,0]; %% Pauli Matrix y
sz=[1,0;0,-1]; %% Pauli Matrix z
n_Pu=[1,0;0,0];    %% Projector onto the excited state Pu=(sz+id)/2;

% % % create initial data binary tree
[X,tau] = init_spin_all_dim_diff_rank(r,2,d);
X = rounding(X,tau);
X = truncate(X,10^-10,30,2);
% [X_test,tau_test] = init_spin_all_dim_diff_rank1(1,d);
% [X,tau] = init_spin_all_dim_diff_rank2(d);
% X = rounding(X,tau);
% [X,tau] = init_spin_all_dim_rank2(r,2,d);

%% construction of the TTNO
% interaction matrix - single particle
A_single = cell(1,d);
for ii=1:d
    A_single{ii} = Omega*sx + Delta*n_Pu;
end
V_single = eye(d,d);

% interaction matrix - long-range interactions
V_int = zeros(d,d);
A_int = cell(1,d);
for ii=1:d
    for jj=1:d
        if ii ~= jj
            V_int(ii,jj) = nu*(1/abs(ii-jj))^alpha;
        end
    end
    A_int{ii} = n_Pu;
end

% HSS 
H_single = hss(V_single,'cluster',1:d);
H_single = adjust_H(H_single);

H_int = hss(V_int,'cluster',1:d);
H_int = adjust_H(H_int);

k = hssrank(H_int); 

% construction of TTNO with help of HSS
TTNO_single = TTNO_HSS_construct_single(H_single,A_single,l,l,n*ones(1,d),1:d);
TTNO_int = TTNO_HSS_construct(H_int,A_int,l,l,n*ones(1,d),1:d);

A = Add_TTN(TTNO_single,TTNO_int,tau);
A = rounding(A,tau);
A{end} = -1i*A{end};

%% make operator magnetisation 
B2 = cell(d,d);
for ii=1:d
    B2{ii,ii} = sz;
end
Mag = make_operator(X,B2,tau,2*ones(d,1));

%% time evolution
time = [];
obs_sz = [];
obs_sz_ad = [];
% obs_sz_BUG = [];

for kk=1:num_runs
    dt = T_end/2^(kk+2);
    dt_save(kk) = dt;
    %%% time-step
    Iter=T_end/dt;
    X_start = X;
    X_start_ad = X;

    tic
    for it=1:Iter
        t0 = (it-1)*dt;
        t1 = it*dt;
        time(it) = t1;
        
%         % time integration with parallel integrator
%         %     X_new = TTN_integrator_complex_parallel_nonglobal(tau,X_start,@F_Ising,t0,t1,A,d,r_min);
%         X_new = TTN_integrator_complex_parallel_nonglobal(tau,X_start,@F_Ising,t0,t1,A,d,r_min);
%         X_new = truncate(X_new,tol,r_max,r_min);

        rej = 1;
        while rej == 1
            [X_new,U_tilde,~] = TTN_integrator_complex_parallel_nonglobal(tau,X_start,@F_Ising,t0,t1,A,d,r_min);
            X_new = truncate(X_new,tol,r_max,r_min);
            rej_rk = rejection_check(X_start,X_new);
            U_tilde_SR = U_tilde_TTN(X_start,X_new);
            rej_eta = eta_check(X_start,X_new,U_tilde_SR,F_Ising(t0,X_new,A,d),const,dt,tol);
            if (rej_rk == 1) || (rej_eta == 1)
                X_start = augment_zero(X_start,X_new); % Kann evtl. paralleler BUG nicht mit 0-Zeilen umghen?
                X_start = orthogonalize(X_start);
            else
                rej = 0;
            end
        end

        % rank-adaptive
        X_new_ad = TTN_integrator_complex_rank_adapt_nonglobal_spin(tau,X_start_ad,@F_Ising,t0,t1,A,d,r_min);
        X_new_ad = truncate(X_new_ad,tol,r_max,r_min);
        
        % setting for next time step
        X_start = X_new;
        X_start_ad = X_new_ad;    
    end

    kk
    if d<=8
        [obs_ref,sol] = ref_sol(d,Omega,nu,Delta,alpha,dt,T_end);
        
        sol_ra_BUG = double(full_tensor(X_new_ad));
        sol_par = double(full_tensor(X_new));
        err_ad(kk) = norm(sol - sol_ra_BUG(:));
        err_par(kk) = norm(sol - sol_par(:));
    end
    
end

figure(1)
loglog(dt_save,err_par,'+-','Linewidth',4)
hold on 
loglog(dt_save,err_ad,'o-','Linewidth',4)
% loglog(dt_save,err_BUG,'-.','Linewidth',2.5)
loglog(dt_save,dt_save,'k','Linewidth',3)
loglog(dt_save,dt_save.^2,'k:','Linewidth',3)
grid on 
set(gca,'YMinorGrid','off')
% title('Error','FontSize', 14)
xlabel('Time step size','FontSize', 20)
ylabel('Error','FontSize', 20)
legend('Error Parallel Integrator','Error Rank-adaptive BUG','h','h^2','FontSize', 20)

% save('workspace_error_plot_d4_Tend1_alpha1')

function [obs_exact,psi] = ref_sol(d,Omega,nu,Delta,alpha,dt,T_end)

%%%% dimension
L=d;
h=Omega;

FinalTime=T_end;
Iter=FinalTime/dt;


sx=[0,1;1,0];
sy=[0,-1i;1i,0];
sz=[1,0;0,-1];
n_Pu=[1,0;0,0];

B = linearisation_long_range_unitary_full(L,sx,n_Pu,nu,Delta,Omega,alpha);

H_sys = superkron(L,B);

psi=zeros(2^L,1);
psi(1,1)=1;

tmp2=diag(sparse(ones(2^(L-1),1)));
X{1,1}=kron(sx,tmp2);
X{L,1}=kron(tmp2,sx);

Z{1,1}=kron(sz,tmp2);
Z{L,1}=kron(tmp2,sz);


for i1=2:L-1
   tmp1=diag(sparse(ones(2^(i1-1),1)));
   tmp2=diag(sparse(ones(2^(L-i1),1)));
   X{i1,1}=kron(kron(tmp1,sx),tmp2);
   Z{i1,1}=kron(kron(tmp1,sz),tmp2);
end

Mag=0*X{1,1};
H_X=0*X{1,1};

for i1=1:L
    Mag=Mag+Z{i1,1};
    H_X=H_X+X{i1,1};
end

Vt=expm(-1i*dt*H_sys); % Vt=expm(-1i*dt*H_sys);

for tst=1:Iter
    time(tst)=tst*dt;
    psi=Vt*psi;
    psi=psi/norm(psi);

    p1(tst)=dot(psi,Mag*psi)/L;
    energy(tst)=dot(psi,H_sys*psi)/L;
    
    mag(tst)=dot(psi,Mag*psi)/L;
end


%%%% diagonalization

Ground_State_Energy=min(real(eig(H_sys)))/L;
%%%%% diagonalization of H_eff
[v2,d2]=eig(full(H_sys));
[x2,y2]=sort(diag(d2),'ascend');


 %%% positive eigenvalue
 ind_p=y2(end);


 norm(H_sys*v2(:,ind_p)-d2(ind_p,ind_p)*v2(:,ind_p));
 Ground_state=v2(:,ind_p);
 Ground_state=Ground_state/norm(Ground_state);
 
 
 Mag_Ground_state=dot(Ground_state,Mag*Ground_state)/L;


obs_exact = mag;

end


function [M] = superkron(L,B)
% B is a cell array
[m,n] = size(B);

M = sparse(2^L,2^L);
for jj=1:m
    tmp = 1;
    for ii=n:-1:1
        if isempty(B{jj,ii}) == 1
            tmp = kron(tmp,speye(2,2));
        else
            tmp = kron(tmp,B{jj,ii});
        end
    end
    M = M + tmp;
end

end
