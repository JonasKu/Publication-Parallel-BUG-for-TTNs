function [U_tilde] = U_tilde_TTN(Y0,Y1)

m = length(Y0) - 2;
U_tilde = cell(1,m+2);

s2 = size(Y1{end});

core1 = double(tenmat(Y0{end},m+1,1:m)).';
core2 = double(tenmat(Y1{end},m+1,1:m)).';

[~,l1] = size(core1);

core2 = core2(:,l1+1:end);
[~,s2(end)] = size(core2);
U_tilde{end} = tensor(mat2tens(core2.',s2,m+1,1:m),s2);

for ii=1:m
    if 1==iscell(Y0{ii})
        U_tilde{ii} = U_tilde_TTN(Y0{ii},Y1{ii});
    else
        s = size(Y0{ii});
        tmp = Y1{ii};
        U_tilde{ii} = tmp(:,s+1:end);
    end
end

end