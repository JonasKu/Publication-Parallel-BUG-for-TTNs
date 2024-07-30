function [Y0] = augment_zero(Y0,Y1)

% augment C_\tau
s1 = size(Y0{end});
s2 = size(Y1{end});
ss = s2 - s1;

if sum(ss) ~= 0
    Y0{end} = augment_zeros_C(Y0{end},s2);
else
    1;
end

% augment the subtrees
m = length(Y0) - 2;
for ii=1:m
    if 1 == iscell(Y0{ii})
        Y0{ii} = augment_zero(Y0{ii},Y1{ii}); % recursion for augmentation
    else
        [~,r1] = size(Y0{ii});
        tmp = Y1{ii};
        Y0{ii} = [Y0{ii} tmp(:,r1+1:end)]; % put bigger basis at leaves
    end     
end

end

function [C] = augment_zeros_C(C,s)

len = length(size(C));
if len == 2
    C(s(1),s(2)) = 0;
elseif len == 3
    C(s(1),s(2),s(3)) = 0;
elseif len == 4
    C(s(1),s(2),s(3),s(4)) = 0;
elseif len == 5
    C(s(1),s(2),s(3),s(4),s(5)) = 0;
elseif len == 6
    C(s(1),s(2),s(3),s(4),s(5),s(6)) = 0;
end

end