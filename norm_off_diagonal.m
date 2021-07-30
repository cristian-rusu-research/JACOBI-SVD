function y = norm_off_diagonal(A)
y = norm(A-diag(diag(A)), 'fro')^2;