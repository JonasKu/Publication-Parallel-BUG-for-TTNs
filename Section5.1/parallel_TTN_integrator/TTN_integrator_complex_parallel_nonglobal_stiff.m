function [Y1,U_tilde,U_tilde_SR] = TTN_integrator_complex_parallel_nonglobal_stiff(tau,Y0,F_tau,t0,t1,A,d,r_min,l_basis)
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
U_tilde = cell(1,m+2);
U_tilde_SR = cell(1,m+2);

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
%          Y1_i = RK_4_nonglobal(Y0_i,tau{i},F_tau_i,t0,t1,A,d);
        Y1_i = Lanczos_matrix(Y0_i,F_tau_i,t0,t1,A,d,l_basis);
    else % if \tau_i \notin L
        [Y1_i,U_tilde{i},U_tilde_SR{i}] = TTN_integrator_complex_parallel_nonglobal_stiff(tau{i},Y0_i,F_tau_i,t0,t1,A,d,r_min,l_basis);
    end
    
    % distiguish between leaf and TTN case
    if 1 == iscell(Y1_i)
        % build up C0_tau_i_hat
        m2 = length(Y1_i) - 2;
        C0_taui_hat = C_zero_augment(Y1_i,Y0,i);
        
        % augment C_tau in 0-direction
        core1 = double(tenmat(C0_taui_hat,m2+1,1:m2)).';
        core2 = double(tenmat(Y1_i{end},m2+1,1:m2)).';
        rr = size(C0_taui_hat);
        
        % use Gram Schmidt/qr for orthonormalization
%         [W_taui_hat,~] = mgson([core1 core2]); 
        [W_taui_hat,~] = qr([core1 core2],0); % qr decomposition   
        [~,s] = size(core1);
        W_taui_hat(:,1:s) = core1; % make sure the first colums are U_0
        r_W = rank([core1 core2]);
        W_taui_hat = W_taui_hat(:,1:r_W); % just take the linear independent columns
        
%        %%% orth check #FIXME
%        tmp1 = core1;
%        tmp2 = W_taui_hat(:,s+1:end);
%        ortho = tmp2'*tmp1;
%        norm(ortho,'Fro');
%        if norm(ortho) > 10^-14
%            1;
%        end 
%        %%%%
        
        % set core tensor of U_tilde for recursion
        M = W_taui_hat(:,s+1:end);
        sz = size(Y1_i{end});
        [~,sz2] = size(W_taui_hat);
        sz(end) = sz2 - s;
        U_tilde{i}{end} = mat2tens(M.',sz,m2+1);
        U_tilde{i}{end} = tensor(U_tilde{i}{end},sz); 
        U_tilde{i}{end-1} = eye(sz(end),sz(end));
       
        U_tilde_SR{i}{end} = U_tilde{i}{end}; % new

        % retensorize
        [~,rr(end)] = size(W_taui_hat);
        Y1_i{end} = tensor(mat2tens(W_taui_hat.',rr,m2+1),rr);
        Y1{i} = Y1_i;
        Y1{end-1} = eye(rr(end),rr(end));

    else
        [~,rr] = size(Y0{i});
        [Y1_i,~] = qr([Y0{i} Y1_i],0); 
        Y1_i(:,1:rr) = Y0{i}; % make sure U0_i is in the right position
        
        Y1{i} = Y1_i;
        U_tilde{i} = Y1_i;

        U_tilde_SR{i} = Y1_i(:,rr+1:end); % new
    end
    
end

%% subflow \Psi
% solve the tensor ODE
C0 = Y0{end};

F_ODE = @(C0,F_tau,U0_tau,t0,A,d) func_ODE(C0,F_tau,Y0(1:m),t0,A,d);

Y1{end-1} = eye(size(Y0{end-1}));
% C1_bar = RK_1_tensor(C0,F_ODE,Y0(1:m),F_tau,t0,t1,tau,A,d);
% C1_bar = RK_4_tensor_nonglobal(C0,F_ODE,Y0(1:m),F_tau,t0,t1,tau,A,d);
C1_bar = Lanczos_tensor(C0,F_ODE,Y1(1:m),F_tau,t0,t1,A,d,l_basis);

%% augmentation
rr = size(C1_bar);
ss = rr;
C1_hat = C1_bar;
F = F_tau(t0,Y0,A,d);
F{end} = (t1-t0)*F{end};

% check in which dimension and how much we need to augment \bar{C}_tau
aug = ones(1,m);
aug(2,1:m) = zeros(1,m);
for jj=1:m
    if 0 == iscell(Y1{jj})
        [~,s1] = size(Y1{jj});
        [~,s2] = size(Y0{jj});
        if s1 == s2 
            aug(1,jj) = 0;
        else
            aug(2,jj) = s1 - s2;
        end
    elseif 1 == iscell(Y1{jj})
        s1 = size(Y1{jj}{end});
        s2 = size(C1_hat);
        if s1(end) == s2(jj)
            aug(1,jj) = 0;
        else
            aug(2,jj) = s1(end) - s2(jj);
        end
    end
end

% augment \bar{C}_tau
for jj=1:m
    % compute Ci - only if there is really an augmentation
%     U_tilde{end} = C1_bar;
    if aug(1,jj) ~= 0 
        tmp = Y0(1:m);
        
        % determine U_tilde
        if 0==iscell(U_tilde{jj})
            tmp2 = U_tilde{jj};
            [~,sz] = size(Y0{jj});
            tmp{jj} = tmp2(:,sz+1:end); % leave case - take only the U_tilde part of \hat{U}
        else
            tmp{jj} = U_tilde{jj}; % TTN case - take the full U_tilde
        end
        
        % multipl. U0 and U_tilde_j to F(Y0)
        dum = cell(1,m);
        for i=1:m
            dum{i} = Mat0Mat0(tmp{i},F{i});
        end
        Ci = ttm(F{end},dum,1:m);

        % augmentation 
        vv = 1:(m+1);
        vv = vv(vv~=jj);
        tmp = double(tenmat(C1_hat,jj,vv));

        mat_Ci = double(tenmat(Ci,jj,vv));
        s_Ci = size(Ci);
        s_Ci = prod(s_Ci(vv));
        tmp(rr(jj)+1:rr(jj)+aug(2,jj),1:s_Ci) = mat_Ci;
        ss(jj) = ss(jj) + aug(2,jj);
        C1_hat = tensor(mat2tens(tmp,ss,jj),ss);
        
    end
    % set core tensor of U_tilde 
    if iscell(U_tilde{jj}) == 1
        U_tilde{jj}{end} = Y1{jj}{end};
    end
end
Y1{end} = C1_hat;
Y1{end-1} = eye(ss(end),ss(end));

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

function [C0_taui_hat] = C_zero_augment(Y1_i,Y0,i)

C0_taui_hat = Y0{i}{end};
tmp = sum(size(C0_taui_hat) == size(Y1_i{end}));
s = size(Y1_i{end});
if tmp < length(size(C0_taui_hat))
    if length(size(C0_taui_hat)) == 2
        C0_taui_hat(s(1),s(2)) = 0;
    elseif length(size(C0_taui_hat)) == 3
        C0_taui_hat(s(1),s(2),s(3)) = 0;
    elseif length(size(C0_taui_hat)) == 4
        C0_taui_hat(s(1),s(2),s(3),s(4)) = 0;
    elseif length(size(C0_taui_hat)) == 5
        C0_taui_hat(s(1),s(2),s(3),s(4),s(5)) = 0;
    end
end

end

