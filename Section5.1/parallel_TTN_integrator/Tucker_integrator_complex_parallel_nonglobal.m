function [Y1,C0_tau_hat] = Tucker_integrator_complex_parallel_nonglobal(tau,Y0,F_tau,t0,t1,A,d,r_min)
% This function does one time-step with the unconventional integrator for
% Tucker tensors in a recursuive way.
%
% Input:
%       tau = representation of the tree
%       Y0 = Tucker; initial value of the integration
%       F_tau = function of the ODE
%       t0,t1 = t1 - t0 is the timestep-size
% Output:
%       Y1 = Tucker; solution of ODE at time t1

Y1 = cell(size(Y0));
m = length(Y0) - 2;
M = cell(1,m);
M2 = cell(1,m);
ten = cell(1,m);
aug = zeros(1,m);

% parfor i=1:m kann parallelisieren, wenn keine Rekursionen vorkommen
for i=1:m
    %% subflow \Phi_i
    v = 1:m+1;
    v = v(v~=i);
    
    Mat_C = tenmat(Y0{end},i,v);
    [Q0_i,S0_i_T] = qr(double(Mat_C).',0);
    
    % K-step
    Y0_i = Ytau_i(tau{i},Y0{i},S0_i_T.'); % initial value for K-step
    
    F_tau_i = @(t,Y_tau_i,A,d) restriction(...
        F_tau(t,prolongation(Y_tau_i,Y0,i,Q0_i),A,d),Y0,i,Q0_i);
    
    Y1_i = RK_4_nonglobal(Y0_i,tau{i},F_tau_i,t0,t1,A,d);
    
    tmp = [Y0{i} Y1_i];
    [U_hat,~] = qr(tmp,0);
    [~,rr] = size(Y0{i});
    U_tilde = U_hat(:,(rr+1):end);
    
    s = size(U_tilde);
    if s(2) == 0
        U_tilde = U_hat; 
        aug(i) = 1;
        Y1{i} = U_tilde;
    else
        Y1{i} = [Y0{i} U_tilde];
    end
    
    M{i} = Mat0Mat0(U_tilde,Y1_i)*Q0_i';
    M2{i} = Mat0Mat0(U_hat,Y0{i});
    ten{i} = tensor(mat2tens(M{i},size(Y0{end}),i));
    
end


%% subflow \Psi
% solve the tensor ODE
C0 = Y0{end};
C0_tau_hat = ttm(C0,M2,1:m);

F_ODE = @(C0,F_tau,U0_tau,t0,A,d) func_ODE(C0,F_tau,Y0(1:m),t0,A,d);

Y1{end-1} = eye(size(Y0{end-1}));
Y1{end} = RK_4_tensor_nonglobal(C0,F_ODE,Y0(1:m),F_tau,t0,t1,tau,A,d); 
% C1_tau_hat = Y1{end};

%% augmentation
C_hat = Y1{end};

for ii=1:m
    if aug(ii) == 0
        tmp1 = tenmat(C_hat,ii);
        tmp2 = tenmat(ten{ii},ii);
        if ii>1
            s1 = size(tmp1);
            s2 = size(tmp2);
            s = s1-s2;
            s2(2) = s(2);
            tmp3 = zeros(s2);
        else
            tmp3 = [];
        end
        tmp = [double(tmp1);[double(tmp2) tmp3]];
        
        r = size(C_hat);
        if r(end) == 1
            r = r(1:end-1);
        end
        r(ii) = 2*r(ii);
        C_hat = mat2tens(tmp,r,ii);
    end
end
s = size(double(C_hat));
Y1{end} = tensor(double(C_hat),[s 1]);


end

function [X] = func_ODE(C,F_tau,U1,t,A,d)
% function [X] = func_ODE(C,F_tau,U1,t,tau)
% This function defines the function F_tau(C(t)X U_1) X U1^*, for the
% tensor-ODE. Here C(t) is in tucker form, C0 = C X M_i, i.e. the M_i are
% matrices.

% argument of F_tau
m = length(U1);
s = size(C);
N = cell(1,m+2);
N{end} = C;
N{end-1} = eye(s(end),s(end));
N(1:m) = U1;

% apply F_tau
% F = F_tau(t,N,tau);
F = F_tau(t,N,A,d);

% multipl. with U1^*
dum = cell(1,m);
for i=1:m
    dum{i} = Mat0Mat0(U1{i},F{i});
end
X = ttm(F{end},dum,1:m);


end