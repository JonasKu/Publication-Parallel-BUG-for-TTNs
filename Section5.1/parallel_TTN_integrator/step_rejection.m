function[rej,X0_aug] = step_rejection(X,X0,F_X0,U_tilde,r_max,tol,h,const)

% If rej == 1 then, we redo the step with the augmented basis;
% If rej == 0 then we accept the step

X0_aug = X; % augmented basis but S0 as core tensors

m = length(X) - 2;
rej_sum = zeros(m,1);
r_C = size(X{end});
r_C0 = size(X0{end});

% build augmented S0 filled with zeros
S0_aug = augment_S0(X0{end},r_C0,r_C);
X0_aug{end} = tensor(double(S0_aug),r_C);
    
for ii=1:m
    if (r_C(ii) == 2*r_C0(ii)) && (r_C(ii) < r_max)
     % repeat step with bigger basis
        rej_sum(ii) = 1;
    end
    
    if iscell(X{ii})
        rej_sum(ii) = step_rejection(X{ii},X0{ii},F_X0{ii},U_tilde{ii},r_max,tol,h,const);
    end

end
if sum(rej_sum) > 0
    rej = 1;
else
    rej = 0;
end

% \eta condition
if rej == 0 % only need to check eta if ranks are not doubled 
    tmp = cell(1,m);
    for ii=1:m % wie mache ich das wenn iwo in U_tilde_SR ein leeres Array steht!?
        tmp{ii} = Mat0Mat0(U_tilde{ii},F_X0{ii});
    end
    ten = ttm(F_X0{end},tmp,1:m);
    tmp = double(tenmat(ten,m+1,1:m));
    eta = norm(tmp,'Fro');
    if h*eta > const*tol 
        rej = 1; % if =1 then I repead the step over and over again as wrong U_tilde is used
    end
end


end


function[S_aug] = augment_S0(S,r0,r1)

S_aug = S;
len = length(r0);

if len == 3
    S_aug(r1(1),r1(2),r1(3)) = 0;
elseif len == 4
    S_aug(r1(1),r1(2),r1(3),r1(4)) = 0;
elseif len == 5
    S_aug(r1(1),r1(2),r1(3),r1(4),r1(5)) = 0;
elseif len == 6
    S_aug(r1(1),r1(2),r1(3),r1(4),r1(5),r1(6)) = 0;
end

end