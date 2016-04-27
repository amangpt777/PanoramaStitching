function dest_pts_nx2 = applyRotationalMatrix(R_3x3, src_pts_nx2, f0, f1)
xs = src_pts_nx2(:,1);
ys = src_pts_nx2(:,2);
n = numel(xs);

column1 = ones(n,1) * f0;
A = [xs ys column1] * R_3x3';
B = [A(:,1)./A(:,3).* f1 A(:,2)./A(:,3).* f1];

dest_pts_nx2 = B;

