function [positions1, values1, Ubar, approx_error, tus] = svd_approximation_two_side_random(U, p, g)
%% Demo code for paper Fast PCA projections by generalized Givens transformations

%% Input:
% U - an orthonormal matrix of size dxd
% g - the number of generalized Givens transformations to use for the approximation

%% Output:
% The g generalized Givens transformations:
% positions - the two indices (i,j) where the transformation operates
% values - the four values of the transformations
% approx_error - the approximation error, as defined in the paper
% tus - the total running time

tic;
[d1, d2] = size(U);
d = max([d1, d2]);

%% basic sanity check
if (d <= 1) || (g < 1)
    positions1 = []; values1 = [];% values2 = [];
    tus = toc;
    return;
end
% if norm(U'*U - eye(d)) >= 10e-7
%     error('U has to be orthogonal');
% end

%% make sure we have a positive integer
g = round(g);

%% vector that will store the indices (i,j) and the values of the transformations for each of the g Givens transformations
positions1 = zeros(2, g);
values1 = zeros(4, g);
% values2 = zeros(4, g);

%% initialization of each Givens transformation
approx_error = [];
for kk = 1:g
    %% check where the maximum scores is, to find the optimum indices
    i_nuc = randsample(1:p, 1);
    if (mod(kk, 2) ~= 0)
        j_nuc = randsample(1:d2, 1);
    else
        j_nuc = randsample(1:d1, 1);
    end

    %% compute the optimum orthogonal transformation on the optimum indices
    if (j_nuc > d1)
        [Uu, ~, Vv] = svd([U(i_nuc, i_nuc) U(i_nuc, j_nuc); 0 0]);
%         Uu = eye(2);
%         x = [U(i_nuc, i_nuc); U(i_nuc, j_nuc)];
%         x = x/norm(x);
%         Vv = [x(1) -x(2); x(2) x(1)];
    else
        [Uu, ~, Vv] = svd(U([i_nuc j_nuc], [i_nuc j_nuc]));
    end

    %% save the Givens transformation
    positions1(1, kk) = i_nuc;
    positions1(2, kk) = j_nuc;
    values1(:, kk) = vec(Uu);
    
    %% update the working matrix
    U = applyGTransformOnRightTransp(U, i_nuc, j_nuc, vec(Vv'));
    if (j_nuc > d1)
%         U = applyGTransformOnLeft_Overcomplete(U, i_nuc, j_nuc, values1(:, kk));
    else
        U = applyGTransformOnLeft(U, i_nuc, j_nuc, values1(:, kk));
    end
    
    approx_error = [approx_error trace(U(1:p, 1:p))];
    stop = 1;
end

%% the explicit approximation, can be avoided
Ubar = eye(d1, d1);% Vbar = eye(d);
for k = 1:g
%     if norm(values1(:, k) - [1; 0; 0; 1]) > 10e-10
    if (positions1(2, k) > d1)
%         Ubar = applyGTransformOnLeft_Overcomplete(Ubar, positions1(1, k), positions1(2, k), values1(:, k));
    else
        Ubar = applyGTransformOnLeft(Ubar, positions1(1, k), positions1(2, k), values1(:, k));
    end
%     end
%     Vbar = applyGTransformOnRightTransp(Vbar, positions1(1, k), positions1(2, k), values2(:, k));
end
% approx_error = [approx_error 1/2*norm(eye(d) - Ubar*U*Vbar,'fro')^2/norm(U,'fro')^2];

%% time everything
tus = toc;
