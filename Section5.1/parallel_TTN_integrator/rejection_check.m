function [rej] = rejection_check(Y0,Y1)

rej = 0;
s1 = size(Y0{end});
s2 = size(Y1{end});

m = length(Y0) - 2;
for ii=1:m
    % check ranks at core tensor
    if s2(ii) == 2*s1(ii)
        rej = 1;
    end
    
    % check ranks at subtrees
    if 1==iscell(Y0{ii})
        rej = rejection_check(Y0{ii},Y1{ii});
    end
end


end