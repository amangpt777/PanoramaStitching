function [ finalImg ] = mergeBlendImgs( imgs, mask, transforms, newHeight, newWidth )
nChannels = size(imgs, 3);
nImgs = size(imgs, 4);

% alpha mask
mask = imcomplement(mask);
mask(1, :) = 1;
mask(end, :) = 1;
mask(:, 1) = 1;
mask(:, end) = 1;
mask = bwdist(mask, 'euclidean');
mask = mask ./ max(max(mask));

% backward transformation
backTransforms = zeros(size(transforms));
for i = 1 : nImgs
    backTransforms(:, :, i) = inv(transforms(:, :, i));
end

% image merging
imgs = double(imgs);
finalImg = zeros([newHeight newWidth nChannels]);
newImg = zeros([newHeight newWidth nChannels], 'double');
alpha = zeros(newHeight, newWidth);
alphaSum = zeros(newHeight, newWidth);
h = 1:newHeight; w = 1:newWidth;
[H, W] = ndgrid(h,w);  % H: mxn, W: mxn
HH = reshape(H,1,[]); WW = reshape(W,1,[]); % column-wise
HWZ = [HH; WW; ones(1, newHeight * newWidth) ];

for k= 1 : nImgs
    % inv(H) * [xd, yd, 1]' = [xs, ys, zs]'
    S =  backTransforms(:, :, k) * HWZ;
    SHW = [S(1,:)./S(3,:); S(2,:)./S(3,:)]; % first row is width, column-segment
    y = reshape(SHW(1, :), newHeight, newWidth) ; % column segment m x n
    x = reshape(SHW(2, :), newHeight, newWidth) ;
    
    alpha(:,:) = interp2(double(mask(:, :)), x, y, 'linear');
    nan_index = find(isnan(alpha));
    if (~isempty(nan_index))
        alpha(nan_index) = 0;
    end
    
    alphaSum = alphaSum + alpha;
    
    newImg(:, :, 1) = interp2(imgs(:, :, 1, k) .* mask, x, y, 'linear');
    newImg(:, :, 2) = interp2(imgs(:, :, 2, k) .* mask, x, y, 'linear');
    newImg(:, :, 3) = interp2(imgs(:, :, 3, k) .* mask, x, y, 'linear');
    nan_index = find(isnan(newImg));
    if (~isempty(nan_index))
        newImg(nan_index) = 0;
    end
    finalImg = finalImg + newImg;
    
end

finalImg = uint8(finalImg ./ (repmat(alphaSum, [1,1,3])));
