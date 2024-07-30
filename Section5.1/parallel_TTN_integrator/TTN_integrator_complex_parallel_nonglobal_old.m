function [Y1,C0_tau_hat,U_aug] = TTN_integrator_complex_parallel_nonglobal(tau,Y0,F_tau,t0,t1,A,d,r_min)
% This function does one time-step with the parallel integrator for
% TTNs in a recursuive way.
%
% Input:
%       tau = representation of the tree
%       Y0 = TTN; initial value of the integration
%       F_tau = function of the ODE
%       t0,t1 = t1 - t0 is the timestep-size
% Output:
%       Y1 = TTN; solution of ODE at time t1

Y1 = cell(size(Y0));
m = length(Y0) - 2;
M = cell(1,m);
U_tilde = cell(1,m+2);
U_aug = cell(1,m+2);

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
    
    if 0 == iscell(tau{i})    % if \tau_i = l, l \in L
         Y1_i = RK_4_nonglobal(Y0_i,tau{i},F_tau_i,t0,t1,A,d);
    else % if \tau_i \notin L
        [Y1_i,C0_taui_hat,U_tilde{i}] = TTN_integrator_complex_parallel_nonglobal(tau{i},Y0_i,F_tau_i,t0,t1,A,d,r_min);
    end
    
    % distiguish between leaf and TTN case
    if 1 == iscell(Y1_i)
        m2 = length(Y1_i) - 2;
        core1 = double(tenmat(C0_taui_hat,m2+1,1:m2)).';
        core2 = double(tenmat(Y1_i{end},m2+1,1:m2)).';
        rr = size(C0_taui_hat);
        
        % natural way to do it 
        [W_taui_hat,~,~] = svd([core1 core2],0);
        [~,s] = size(core1);
        W_taui_hat(:,1:s) = core1;
        W_taui_hat = W_taui_hat(:,1:rank(W_taui_hat));

        [row,rr_hat] = size(W_taui_hat);
        if rr_hat < 2*rr(end)
            W_taui_hat(:,rr_hat+1:2*rr(end)) = zeros(row,2*rr(end)-rr_hat);
        end

%         % test - augment with zeros after core1.
%         [W_taui_hat,~,~] = svd([core1 core2],0);
%         [row,rr_hat] = size(W_taui_hat);
%         if rr_hat < 2*rr(end)
%             [~,s] = size(core1);
%             r_core1 = rank(core1);
%             tmp = core1;
%             tmp(:,r_core1+1:s) = zeros(row,s-r_core1);
%             tmp(:,s+1:2*rr(end)) = W_taui_hat(:,s+1:end);
%             W_taui_hat = tmp;
%         else
%             [~,s] = size(core1);
%             W_taui_hat(:,1:s) = core1;
%         end

        % retensorize
        rr(end) = 2*rr(end);
        Y1_i{end} = tensor(mat2tens(W_taui_hat.',rr,m2+1),rr);
        Y1{i} = Y1_i;
        
    else
        [~,rr] = size(Y0{i});
        [Y1_i,~] = qr([Y0{i} Y1_i],0); 
        Y1_i(:,1:rr) = Y0{i}; % make sure U0_i is in the right order
        [row,rr_hat] = size(Y1_i);
        
        if rr_hat < 2*rr
            Y1_i(:,rr_hat+1:2*rr) = zeros(row,2*rr-rr_hat);
        end
        Y1{i} = Y1_i;
        
        % \tilde{U}_\taui^1
        tmp = Y1{i};
        U_tilde{i} = tmp(:,(rr+1):end); 
        U_aug{i} = Y1_i;
%         Y1{i} = Y1_i(:,1:rr_hat);
    end
    M{i} = Mat0Mat0(Y1_i,Y0{i});
    
end

%% subflow \Psi
% solve the tensor ODE
C0 = Y0{end};

F_ODE = @(C0,F_tau,U0_tau,t0,A,d) func_ODE(C0,F_tau,Y0(1:m),t0,A,d);

Y1{end-1} = eye(size(Y0{end-1}));
% C1_bar = RK_1_tensor(C0,F_ODE,Y0(1:m),F_tau,t0,t1,tau,A,d);
C1_bar = RK_4_tensor_nonglobal(C0,F_ODE,Y0(1:m),F_tau,t0,t1,tau,A,d);
U_tilde{end} = C1_bar;
C0_tau_hat = ttm(Y0{end},M,1:m);


% augmentation
rr = size(C1_bar);
ss = rr;
C1_hat = C1_bar;
F = F_tau(t0,Y0,A,d);
F{end} = (t1-t0)*F{end};
for jj=1:m
    % compute Ci
    tmp = Y0(1:m);
    tmp{jj} = U_tilde{jj};
    % multipl. U0 and U_tilde_j
    dum = cell(1,m);
    for i=1:m
        dum{i} = Mat0Mat0(tmp{i},F{i});
    end
    Ci = ttm(F{end},dum,1:m);
    
    % augmentation
    vv = 1:(m+1);
    vv = vv(vv~=jj);
    tmp = double(tenmat(C1_hat,jj,vv));
    
    s_tmp = size(Ci);
    s_tmp = prod(s_tmp(vv));
    tmp(rr(jj)+1:2*rr(jj),1:s_tmp) = double(tenmat(Ci,jj,vv));
    ss(jj) = 2*ss(jj);
    C1_hat = tensor(mat2tens(tmp,ss,jj),ss);
end
Y1{end} = C1_hat;
U_aug{end} = C1_hat;
% Y1{end} = C1_bar; % test

% U_aug = 1;

end

function [X] = func_ODE(C,F_tau,U0,t,A,d)
% function [X] = func_ODE(C,F_tau,U1,t,tau)
% This function defines the function F_tau(C(t)X U_0) X U0^*, for the
% tensor-ODE. Here C(t) is in tucker form, C0 = C X M_i, i.e. the M_i are
% matrices.

% argument of F_tau
m = length(U0);
s = size(C);
N = cell(1,m+2);
N{end} = C;
N{end-1} = eye(s(end),s(end));
N(1:m) = U0;

% apply F_tau
% F = F_tau(t,N,tau);
F = F_tau(t,N,A,d);

% multipl. with U0^*
dum = cell(1,m);
for i=1:m
    dum{i} = Mat0Mat0(U0{i},F{i});
end
X = ttm(F{end},dum,1:m);


end

