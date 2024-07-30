function [Y1,C0_tau_hat,U_tilde] = TTN_integrator_parallel_2nd_order_nonglobal(tau,Y0,F_tau,t0,t1,A,d,r_min)
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
Y0_save = cell(1,m);
% M = cell(1,m);
Mi = cell(1,m);
U_tilde = cell(1,m+2);
rr_save = zeros(1,m);

% pre-step
U0_hat = cell(1,m);
F = F_tau(t0,Y0,A,d);
for jj=1:m
    v = 1:m+1;
    v = v(v~=jj);
    Mat_C = tenmat(Y0{end},jj,v);
    [Q0_i,~] = qr(double(Mat_C).',0);
    
    % computes the U0_hat
    tmp = 1;
    for ii=1:m
        if ii ~= jj
            tmp = kron(Mat0Mat0(conj(F{ii}),Y0{ii}),tmp); % conj as we want ^T, not ^*
        end
    end
    mat = tenmat(F{end},jj,v);
    [U0_hat{jj},~] = qr(F{jj}*(double(mat)*(tmp*Q0_i)),0);
end
U0_hat{end+2} = Y0{end};
U0_hat{end-1} = 1;
for jj=1:m
    tmp = 1;
    for ii=1:m
        if ii ~= jj
            tmp = kron(Mat0Mat0(Y0{ii},U0_hat{ii}),tmp);
        end
    end
    Mi{jj} = Q0_i.'*tmp*Q0_i;
end


% parfor i=1:m kann parallelisieren, wenn keine Rekursionen vorkommen
for i=1:m
    %% subflow \Phi_i
    v = 1:m+1;
    v = v(v~=i);
    
    Mat_C = tenmat(Y0{end},i,v);
    [Q0_i,S0_i_T] = qr(double(Mat_C).',0);
    
    % K-step
    Y0_i = Ytau_i(tau{i},Y0{i},S0_i_T.'); % initial value for K-step
    Y0_i = Y0_i*Mi{i};
    
    F_tau_i = @(t,Y_tau_i,A,d) restriction(...
        F_tau(t,prolongation(Y_tau_i,U0_hat,i,Q0_i),A,d),U0_hat,i,Q0_i);
    
    if 0 == iscell(tau{i})    % if \tau_i = l, l \in L
         Y1_i = RK_4_nonglobal(Y0_i,tau{i},F_tau_i,t0,t1,A,d);
%          Y1_i = RK_1(Y0_i,tau{i},F_tau_i,t0,t1,A,d);
    else % if \tau_i \notin L
        [Y1_i,C0_taui_hat,U_tilde{i}] = TTN_integrator_parallel_2nd_order_nonglobal(tau{i},Y0_i,F_tau_i,t0,t1,A,d,r_min);
    end
    
    % distiguish between leaf and TTN case
    if 1 == iscell(Y1_i)
        m2 = length(Y1_i) - 2;
        tmp = [double(tenmat(C0_taui_hat,m2+1,1:m2)).' double(tenmat(Y1_i{end},m2+1,1:m2)).'];
        s = size(C0_taui_hat);
        [W_taui_hat,S,~] = svd(tmp,0);
        
        [ss,rr] = size(S);
        
        if 2*s(end) < rr
            tmp = 2*s(end) - rr;
            W_taui_hat = [W_taui_hat zeros(ss,tmp)];
        end
        s(end) = 2*s(end);
        Y1_i{end} = tensor(mat2tens(W_taui_hat.',s,m2+1),s);
        Y1{i} = Y1_i;
        
    else
        [~,rr] = size(Y0{i});
        [Y1_i,~] = qr([Y0{i} Y1_i],0); % as in the paper
%         [Y1_i,~] = qr([Y1_i Y0{i}],0); % as in rank-adapt
        [row,rr_hat] = size(Y1_i);
        rr_save(i) = rr_hat;
        
        if rr_hat < 3*rr
            Y1_i(:,rr_hat+1:3*rr) = zeros(row,3*rr-rr_hat);
%             aug(i) = 1;
            Y1{i} = Y1_i;
        else
            Y1{i} = Y1_i;
        end
        
        % \tilde{U}_\taui^1
        tmp = Y1{i};
        [~,rrr] = size(Y0{i});
        U_tilde{i} = tmp(:,(rrr+1):end); % geht nur für nicht TTN
        
    end

    Y0_save{i} = Y0{i};
%     M{i} = Mat0Mat0(Y1_i,Y0{i});
    Mi{i} = Mi{i}'; % in Jonas notes its .', but for complex cases I think we need this
%     M1{i} = Mat0Mat0(Y1_i(:,1:2),Y0{i}); % test
    
end

%% subflow \Psi
% solve the tensor ODE
C0 = ttm(Y0{end},Mi,1:m);
% C0 = Y0{end};

F_ODE = @(C0,F_tau,U0_tau,t0,A,d) func_ODE(C0,F_tau,Y0(1:m),t0,A,d);

Y1{end-1} = eye(size(Y0{end-1}));
% C1_bar = RK_1_tensor(C0,F_ODE,U0_hat(1:m),F_tau,t0,t1,tau,A,d);
C1_bar = RK_4_tensor_nonglobal(C0,F_ODE,U0_hat(1:m),F_tau,t0,t1,tau,A,d);
U_tilde{end} = C1_bar;
C0_tau_hat = 1; % ttm(Y0{end},M,1:m);

% augmentation
rr = size(C1_bar);
ss = rr;
C1_hat = C1_bar;
F = F_tau(t0,Y0,A,d);
F{end} = (t1-t0)*F{end};
for jj=1:m
    % compute Ci
    tmp = U0_hat;
    tmp{jj} = U_tilde{jj};
    % multipl. with U1^*
    dum = cell(1,m);
    for i=1:m
        dum{i} = Mat0Mat0(tmp{i},F{i});
    end
    Ci = ttm(F{end},dum,1:m);
    
    % augmentation
    vv = 1:(m+1);
    vv = vv(vv~=jj);
    tmp = double(tenmat(C1_hat,jj,vv));
    % tmp(rr(jj)+1:2*rr(jj),1:rr(jj)) = double(tenmat(Ci,jj,vv));
    
    s_tmp = size(Ci);
    s_tmp = prod(s_tmp(vv));
    tmp(rr(jj)+1:3*rr(jj),1:s_tmp) = double(tenmat(Ci,jj,vv));
    ss(jj) = 3*ss(jj);
    C1_hat = tensor(mat2tens(tmp,ss,jj),ss);
end
Y1{end} = C1_hat;

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

