function [inliers_id, H] = runRANSAC_H(Xs, Xd, ransac_n, eps)
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
    H_3x3 = computeHomography(Xsr, Xdr);
    % apply the homography to all the points
    Xdc = applyHomography(H_3x3, Xs);
    % Euclidean distance (row sum)
    dist = sqrt(sum((Xd-Xdc).^2,2));
    % find all the points with the dist less that threshold
    index = find(dist < eps);
    inlierNum = numel(index);
    if max_inlierNum < inlierNum
        max_inlierNum = inlierNum;
        inliers_id = index;
        H = H_3x3;       
    end
    
end

