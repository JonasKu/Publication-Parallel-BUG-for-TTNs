function [Y1] = parallel_TTN_integrator(tau,Y0,F_tau,t0,t1,A,d,r_min)

% every TTN (Y0 and A) are already stored as a list of nodes
[Y0_vec,num] = vec_TTN(Y0);
A_vec = vec_TTN(A);

len = length(Y0_vec);


parfor ii=1:len
    
end



Y1 = re_vec_TTN(Y0_vec,Y0,num);

end