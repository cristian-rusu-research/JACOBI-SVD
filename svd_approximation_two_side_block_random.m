function [positions1, values1, approx_error, tus] = svd_approximation_two_side_block_random(U, p, b)
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
if (d <= 1) || (b < 1)
    positions1 = []; values1 = [];% values2 = [];
    tus = toc;
    return;
end
% if norm(U'*U - eye(d)) >= 10e-7
%     error('U has to be orthogonal');
% end

%% make sure we have a positive integer
b = round(b);

%% vector that will store the indices (i,j) and the values of the transformations for each of the g Givens transformations
positions1 = zeros(2*p, b);
values1 = zeros(4*p^2, b);
values2 = []; %zeros(4*p^2, b);

%% initialization of each Givens transformation
approx_error = [];
for kk = 1:b
    indices = [1:p randsample(p+1:d, p)];
    indices = unique(sort([1:p indices]));
    
    if (length(indices) < 2*p)
        other_indices = setdiff(1:d1, indices');
        indices = [indices other_indices(1:(2*p-length(indices)))];
        indices = unique(sort(indices));
    end

    %% compute the optimum orthogonal transformation on the optimum indices
    under = indices(find(indices <= d1));
    over = indices(find(indices > d1));
    [Uu, ~, Vv] = svd([U(under, indices); zeros(length(over), length(indices))]);

    %% save the Givens transformation
    positions1(:, kk) = indices;
    values1(1:length(vec(Uu)), kk) = vec(Uu);
%     values1 = vec(Uu(1:length(under), 1:length(under)));
%     values2(:, kk) = vec(Vv);
    
    %% update the working matrix
%     GG = speye(d1);
%     GG(under, under) = reshape(values1, length(under), length(under));
%     U = GG'*U;
    
    U(under(1:length(under)), :) = Uu(1:length(under), 1:length(under))'*U(under(1:length(under)), :);
    
%     GG = speye(d);
%     GG(indices, indices) = reshape(vec(Vv), 2*p, 2*p);
    U(:, indices) = U(:, indices)*Vv;
    
    approx_error = [approx_error trace(U(1:p, 1:p))];
    stop = 1;
end

%% the explicit approximation, can be avoided
% Ubar = eye(d);% Vbar = eye(d);
% for k = 1:g
%     Ubar = applyGTransformOnLeft(Ubar, positions1(1, k), positions1(2, k), values1(:, k));
% %     Vbar = applyGTransformOnRightTransp(Vbar, positions1(1, k), positions1(2, k), values2(:, k));
% end
% approx_error = [approx_error 1/2*norm(eye(d) - Ubar*U*Vbar,'fro')^2/norm(U,'fro')^2];

%% time everything
tus = toc;
