function dest_pts_nx2 = applyHomography(H_3x3, src_pts_nx2)
xs = src_pts_nx2(:,1);
ys = src_pts_nx2(:,2);
n = numel(xs);

column1 = ones(n,1);
A = [xs ys column1] * H_3x3';
B = [A(:,1)./A(:,3) A(:,2)./A(:,3)];

dest_pts_nx2 = B;

