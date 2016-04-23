function R_3x3 = computeRotationalMatrix(src_pts_nx2, dest_pts_nx2, f0, f1)
xs = src_pts_nx2(:,1);
ys = src_pts_nx2(:,2);
xd = dest_pts_nx2(:,1);
yd = dest_pts_nx2(:,2);

n = numel(xs);

column1 = ones(n,1) * f0*f1;
column0 = zeros(n,3);

Ax = [xs*f1 ys*f1 column1 column0 -xd.*xs -xd.*ys -xd*f0];
Ay = [column0 xs*f1 ys*f1 column1 -yd.*xs -yd.*ys -yd*f0];

A = [Ax;Ay];

[V,D] = eig(A'*A); % V is the matrix whose column are the corresponding vectors
% the return eigenvalues is not in sorted order, we must sort it
ev = diag(D);
[ev_sorted, ind] = sort(ev);
% l = norm(V(:,1));  % used to check the ||h|| = 1
R_3x3 = reshape(V(:,ind(1)), [3,3])'; % reshape is column-wise, we want row-wise
                                 % we need to transpose it

end