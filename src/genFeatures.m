function [F, P, F_strongest] = genFeatures(input_imgs)

n_imgs = length(input_imgs);
F = cell(n_imgs, 1);
P = cell(n_imgs, 1);
F_strongest = cell(n_imgs, 1);

for i = 1:n_imgs
    gray = rgb2gray(input_imgs{i});
    points = detectSURFFeatures(gray);
    [f, p] = extractFeatures(gray, points);
    F{i} = f;
    P{i} = p.Location;
    P_strongest = p.selectStrongest(200).Location;
    [~, index, ~] = intersect(P{i}, P_strongest, 'rows');
    F_strongest{i} = F{i}(index, :);
%     F_strongest{i} = F{i};
end
% indexPairs = matchFeatures(F{4},F{5})
% matchedOriginal  = P{4}(indexPairs(:,1), :)
% matchedDistorted = P{5}(indexPairs(:,2), :);
% [H, inlier_pts_distorted, inlier_pts_original, status] = ... 
%             estimateGeometricTransform...
%             (matchedDistorted, matchedOriginal, 'projective');
% figure
% showMatchedFeatures(rgb2gray(input_imgs{4}),rgb2gray(input_imgs{5}),inlier_pts_original,inlier_pts_distorted)
% title('Candidate matched points (including outliers)')

