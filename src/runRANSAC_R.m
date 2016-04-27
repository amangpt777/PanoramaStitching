function [inliers_id, R] = runRANSAC_R(Xs, Xd, ransac_n, eps, f0, f1)
% inlier number
np = size(Xs,1);
max_inlierNum = 0;
s = 4;
for it = 1:ransac_n
    % random choose s samples
    pind = randperm(np,s);
    Xsr = Xs(pind,:);
    Xdr = Xd(pind,:);
    % use the random points to traning the homography model
    R_3x3 = computeRotationalMatrix(Xsr, Xdr, f0, f1);
    % apply the homography to all the points
    Xdc = applyRotationalMatrix(R_3x3, Xs, f0, f1);
    % Euclidean distance (row sum)
    dist = sqrt(sum((Xd-Xdc).^2,2));
    % find all the points with the dist less that threshold
    index = find(dist < eps);
    inlierNum = numel(index);
    if max_inlierNum < inlierNum
        max_inlierNum = inlierNum;
        inliers_id = index;
        R = R_3x3;       
    end
    
end

