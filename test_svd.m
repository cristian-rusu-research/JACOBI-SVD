close all
clear
clc

%% generate random orthogonal matrix
dataset = 'MNIST';
load('MNIST_knn.mat');
X = bsxfun(@minus, X, mean(X));
X = bsxfun(@rdivide, X, sqrt(sum(X.^2)));

p = 50;

d = size(X, 1);
N = size(X, 2);

tic; [U, S, V] = svds(X, p); toc_svd = toc;

%% number of Givens transformations
m = round(10*p*log2(d));

[positions1, values1, values2, Ubar, approx_error, tus] = svd_approximation_two_side(X, p, m);
[positions1_max, values1_max, Ubar_max, approx_error_max, tus_max] = svd_approximation_two_side_max(X, p, m);
[positions1_random, values1_random, Ubar_random, approx_error_random, tus_random] = svd_approximation_two_side_random(X, p, m);

f = figure;
subplot(1, 2, 1);
plot((approx_error)/trace(S)*100, '--b', 'LineWidth', 2);
hold on; plot((approx_error_max)/trace(S)*100, '--r', 'LineWidth', 2);
hold on; plot((approx_error_random)/trace(S)*100, '--g', 'LineWidth', 2);
axis([0 m 0 max([approx_error approx_error_max])/trace(S)*100]);
grid on;
legend('proposed', 'two sided jacobi', 'random', 'Location', 'NorthWest', 'interpreter', 'latex', 'FontSize', 14);
xlabel('Number of transformations $m$', 'interpreter', 'latex', 'FontSize', 14);
ylabel('Accuracy (\%)', 'interpreter', 'latex', 'FontSize', 14);

b = round(m/p/5);
[positions1_block, values1_block, approx_error_block, tus_block] = svd_approximation_two_side_block(X, p, b);
[positions1_block_max, values1_block_max, approx_error_block_max, tus_block_max] = svd_approximation_two_side_block_max(X, p, b);
[positions1_block_random, values1_block_random, approx_error_block_random, tus_block_random] = svd_approximation_two_side_block_random(X, p, b);

subplot(1, 2, 2);
plot((approx_error_block)/trace(S)*100, '-bs', 'LineWidth', 2);
hold on; plot((approx_error_block_max)/trace(S)*100, '-rp', 'LineWidth', 2);
hold on; plot((approx_error_block_random)/trace(S)*100, '-gv', 'LineWidth', 2);
axis([1 b 0 max(approx_error_block)/trace(S)*100+0.1]);
grid on;
legend('block proposed', 'block two sided jacobi', 'block random', 'Location', 'NorthWest', 'interpreter', 'latex', 'FontSize', 14);
xlabel('Number of block transformations $b$', 'interpreter', 'latex', 'FontSize', 14);
ylabel('Accuracy (\%)', 'interpreter', 'latex', 'FontSize', 14);

f.Position = [241 534 950 322];
ha = get(gcf, 'Children');
set(ha(2), 'Position', [0.5303    0.1431    0.3347    0.7819]);
