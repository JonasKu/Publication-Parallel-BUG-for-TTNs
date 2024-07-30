function [rej] = eta_check(Y0,Y1,U_tilde_SR,F_Y0,const,h,tol)
% only for binary trees

rej = 0;
m = length(Y0) - 2;
tmp = cell(1,m);
U_tilde = Y1; % new

for i=1:m
    if 1==iscell(Y0{i})
        U_tilde{i}{end} = U_tilde_SR{i}{end};
        rr = size(U_tilde_SR{i}{end});
        ss = size(F_Y0{i}{end});
        if rr(end) == 0
            tmp{i} = 0*zeros(rr(end),ss(end)); %*Mat0Mat0(U_tilde{i},F_Y0{i});
        else
            tmp{i} = Mat0Mat0(U_tilde{i},F_Y0{i});
        end
    else
        if 1==isempty(U_tilde_SR{i}) 
            tmp{i} = 0*Mat0Mat0(U_tilde{i},F_Y0{i});
        else
            tmp{i} = Mat0Mat0(U_tilde_SR{i},F_Y0{i});
        end
    end
    
    % going to deeper levels
    if 1==iscell(Y0{i})
        v = 1:m+1;
        v = v(v~=i);
        Mat_C = tenmat(Y0{end},i,v);
        [Q0_i,~] = qr(double(Mat_C).',0);
        F_Y0_i = restriction(F_Y0,Y0,i,Q0_i); % restirct F_Y0_i to lower tree
        
        % U_tilde anpasen
        rej = eta_check(Y0{i},U_tilde{i},U_tilde_SR{i},F_Y0_i,const,h,tol);
    end
end


F_prod = double(ttm(F_Y0{end},tmp,[1 2]));
eta = norm(F_prod(:),'Fro');
if h*eta > const*tol
    rej = 1;
end


end