function H_3x3 = computeHomography(src_pts_nx2, dest_pts_nx2)
xs = src_pts_nx2(:,1);
ys = src_pts_nx2(:,2);
xd = dest_pts_nx2(:,1);
yd = dest_pts_nx2(:,2);

n = numel(xs);

column1 = ones(n,1);
column0 = zeros(n,3);

Ax = [xs ys column1 column0 -xd.*xs -xd.*ys -xd];
Ay = [column0 xs ys column1 -yd.*xs -yd.*ys -yd];

A = [Ax;Ay];

[V,D] = eig(A'*A); % V is the matrix whose column are the corresponding vectors
% the return eigenvalues is not in sorted order, we must sort it
ev = diag(D);
[ev_sorted, ind] = sort(ev);
% l = norm(V(:,1));  % used to check the ||h|| = 1
H_3x3 = reshape(V(:,ind(1)), [3,3])'; % reshape is column-wise, we want row-wise
                                 % we need to transpose it

%{
% Solve equations using SVD
X = xs'; Y = ys';
x = xd'; y = yd';
rows0 = zeros(3, n);
rowsXY = -[X; Y; ones(1,n)];
hx = [rowsXY; rows0; x.*X; x.*Y; x];
hy = [rows0; rowsXY; y.*X; y.*Y; y];
h = [hx hy]; % 9x2n
if n == 4
    [U, ~, ~] = svd(h);
else
    [U, ~, ~] = svd(h, 'econ');
end
v = reshape(U(:,9), 3, 3)'
%}
