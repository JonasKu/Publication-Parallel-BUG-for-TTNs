clear all; close all; clc
addpath('C:\Users\Dominik\Documents\MATLAB\Low rank approximations\unconventional_TTN_integrator')
addpath('C:\Users\Dominik\Documents\MATLAB\Low rank approximations\parallel_TTN_integrator')
addpath('C:\Users\Dominik\Documents\MATLAB\Low rank approximations\rank_adaptive_integrator_for_TTN')
addpath('C:\Users\Dominik\Documents\MATLAB\Low rank approximations\TTNO\HSS_and_no_structure_case')
addpath('C:\Users\Dominik\Documents\MATLAB\Matlab toolboxes\hm-toolbox-master')

% number particles
d = 16;
l = log(d)/log(2); % number of layers
n = 2;             % physical dimension

% parameters of the model
nu = 2;
Delta = 1;
Omega = 1;
alpha = 1;
% time step size and final time
T_end = 1;
dt = 0.01;
% rank of initial data at bottom layer
r = 2;
% bond dimension operator
r_op_max = 50;
r_op_min = 2;

% for rank-adaptive integrator
tol = 10^-8;
r_min = 2;
r_max = 40;

% constant for step rejection
const = 10;

%%% time-step
Iter=T_end/dt;

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
obs_sz_BUG = [];

tmp = F_Ising(0,X,A,d);
en = -1i*Mat0Mat0(X,tmp);
nn = sqrt(abs(Mat0Mat0(X,X))); % times i, as we only want the Hamiltonian

en_ad = -1i*Mat0Mat0(X,tmp);
nn_ad = sqrt(abs(Mat0Mat0(X,X))); % times i, as we only want the Hamiltonian

r_max_pa = max_rank(X);
r_max_ad = max_rank(X);

X_start = X;
X_start_ad = X;

tic
for it=1:Iter
    t0 = (it-1)*dt;
    t1 = it*dt;
    time(it) = t1;
    
%     % without step rejection
%     [X_new,~,~] = TTN_integrator_complex_parallel_nonglobal(tau,X_start,@F_Ising,t0,t1,A,d,r_min);
%     X_new = truncate(X_new,tol,r_max,r_min);
    
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
    
%     % time integration 2nd order parallel integrator
%     X_new = TTN_integrator_parallel_2nd_order_nonglobal(tau,X_start,@F_Ising,t0,t1,A,d,r_min);
%     X_new = truncate(X_new,tol,r_max,r_min);
    
    % rank-adaptive
    X_new_ad = TTN_integrator_complex_rank_adapt_nonglobal_spin(tau,X_start_ad,@F_Ising,t0,t1,A,d,r_min);
    X_new_ad = truncate(X_new_ad,tol,r_max,r_min);
 
    % norm and energy 
    en(it+1) = -1i*Mat0Mat0(X_new,F_Ising(0,X_new,A,d)); % times i, as we only want the Hamiltonian
    nn(it+1) = sqrt(abs(Mat0Mat0(X_new,X_new)));
    
    en_ad(it+1) = -1i*Mat0Mat0(X_new_ad,F_Ising(0,X_new_ad,A,d)); % times i, as we only want the Hamiltonian
    nn_ad(it+1) = sqrt(abs(Mat0Mat0(X_new_ad,X_new_ad)));
    
%     % renormalisation
%     X_new{end} = X_new{end}/sqrt(abs(Mat0Mat0(X_new,X_new)));
%     sqrt(abs(Mat0Mat0(X_new,X_new)))

    % setting for next time step
    X_start = X_new; 
    X_start_ad = X_new_ad;

    % compute magnetisation in z-direction
    tmp = apply_operator_nonglobal(X_new,Mag,d);
    obs_sz(it) = (1/d)*Mat0Mat0(X_new,tmp);
    
    tmp = apply_operator_nonglobal(X_new_ad,Mag,d);
    obs_sz_ad(it) = (1/d)*Mat0Mat0(X_new_ad,tmp);
 
    % max rank
    r_max_pa(it+1) = max_rank(X_new);
    r_max_ad(it+1) = max_rank(X_new_ad);
    
    it
    
end
total_time = toc;

if d<=8
    [obs_ref,sol] = ref_sol(d,Omega,nu,Delta,alpha,dt,T_end);
end

figure(1)
if d<=8
    plot(time,obs_ref,'k','Linewidth',1.5)
end
hold on
plot(time,real(obs_sz),'Linewidth',1.5)
plot(time,real(obs_sz_ad),'-.','Linewidth',1.5)
% plot(time,real(obs_sz_BUG),':','Linewidth',1.5)
xlabel('Time')
title('Magnetization in z-direction')
legend('Exact reference solution','Parallel Integrator','Rank-adaptive BUG')

% figure(2)
% plot([0 time],abs(nn),'Linewidth',4)
% hold on
% plot([0 time],abs(nn_ad),'-.','Linewidth',4)
% % plot([0 time],abs(nn_BUG),':','Linewidth',2)
% xlabel('Time')
% title('Norm')
% legend('Parallel Integrator','Rank-adaptive BUG')
% 
% figure(3)
% plot([0 time],abs(en),'Linewidth',4)
% hold on
% plot([0 time],abs(en_ad),'-.','Linewidth',4)
% % plot([0 time],abs(en_BUG),':','Linewidth',2)
% xlabel('Time')
% title('Energy')
% legend('Parallel Integrator','Rank-adaptive BUG')

figure(7)
subplot(1,2,1)
plot([0 time],abs(nn),'Linewidth',4)
hold on
plot([0 time],abs(nn_ad),'-.','Linewidth',4)
% plot([0 time],abs(nn_BUG),':','Linewidth',2)
xlabel('Time')
title('Norm')
legend('Parallel Integrator','Rank-adaptive BUG')
subplot(1,2,2)
plot([0 time],abs(en),'Linewidth',4)
hold on
plot([0 time],abs(en_ad),'-.','Linewidth',4)
% plot([0 time],abs(en_BUG),':','Linewidth',2)
xlabel('Time')
title('Energy')
legend('Parallel Integrator','Rank-adaptive BUG')


figure(4) 
plot([0 time],r_max_pa,'Linewidth',4)
hold on
plot([0 time],r_max_ad,'-.','Linewidth',4)
% plot([0 time],r_max_BUG,':','Linewidth',2)
xlabel('Time')
title('Max. ranks')
legend('Parallel Integrator','Rank-adaptive BUG')

figure(5)
semilogy(time,abs(obs_ref - obs_sz),'Linewidth',4)
hold on
semilogy(time,abs(obs_ref - obs_sz_ad),'Linewidth',4)
legend('Error Parallel BUG','Error Rank-adaptive BUG')

% Error of state at final time
% T = full_tensor(X_new);
% mat = tenmat(T,d+1,1:d);
% mat = double(mat);
% 
% TT = full_tensor(X_new_ad);
% mat_T = tenmat(TT,d+1,1:d);
% mat_T = double(mat_T);
% err = norm(sol - mat(:))
% err = norm(sol - mat_T(:))

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