function [B] = linearisation_long_range_unitary_full(d,sx,n,V,Delta,Omega,alpha)

B = cell(d,d);

for k=1:d
    B{k,k} = Omega*sx + Delta*n;
end

% long range interaction
count = d + 1;
for ii=1:d
    for jj=1:d
        if ii==jj
            
        else
            B{count,ii} = (V/2)*(1/abs(ii-jj)^alpha)*n;
            B{count,jj} = n;
            
            count = count + 1;
        end
    end
end