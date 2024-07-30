function [B_vec,num] = vec_TTN(A)

B_vec = cell(1,1);
B_vec{1,1} = A{end};

m = length(A) - 2;
num = cell(1,m+1);
num{end} = 1;

for ii=1:m
        if iscell(A{ii}) == 1
            [tmp,num{ii}] = vec_TTN(A{ii});
            B_vec = [B_vec tmp];
            num{end} = num{end} + num{ii}{end};
        else
            B_vec = [B_vec A{ii}];
            num{ii} = 1;
            num{end} = num{end} + num{ii};
        end
        
end   


end