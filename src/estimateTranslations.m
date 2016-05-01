%% estimate pairwise translations
%  input:   imgs - source images
%  output:  translations - translation matrices to align each pair of images
function [ translations ] = estimateTranslations( imgs )
% parameters
successProb = 0.99;
inlierRatio = 0.3;
epsilon = 1.5;

% image information
nImgs = size(imgs, 4);

% pairwise translation estimation
translations = zeros(3, 3, nImgs);
translations(:, :, 1) = eye(3);
img_prev = imgs(:, :, :, 1);
for i = 2 : nImgs
    img_curr = imgs(:, :, :, i);
    [locs, locd] = genSIFTMatches(img_prev, img_curr); % first img is source image
    np = size(locs,1);
    matches = zeros(np,3,2);  % matches [col row z=1, 2]
    matches(:,:,1) = [locs(:,2) locs(:,1) ones(np,1)];
    matches(:,:,2) = [locd(:,2) locd(:,1) ones(np,1)];
    translations(:, :, i) = RANSAC(successProb, inlierRatio, 1, matches, epsilon, @computeTranslation, @applyTranslation);
    img_prev = img_curr;
end
end

