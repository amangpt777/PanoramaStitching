function [ newImg ] = createCylPanorama( imgs, loop)

% estimate focal length
ransac_n = 50; % Max number of iteractions
ransac_eps = 2; % Acceptable alignment error % 3 pixel far
focal_length_type = 'variable';
[xs, xd] = genSIFTMatches(imgs(:,:,:,2), imgs(:,:,:,1));
focal = zeros(1, 10);
for i = 1:10
    while (1)
        [inliers_id, H_3x3] = runRANSAC_H(xs, xd, ransac_n, ransac_eps);
        [f0, f1] = computeFocalLength(imgs(:,:,:,2), H_3x3, focal_length_type);
        if isreal(f0) && isreal(f1)
            break
        end
    end
    focal(i) = sqrt(f0*f1);
end

f = mean(focal);
%f = 2500;
%fprintf('focal length is %12.4f \n', f);
k1 = -0.15; %radial distortion parameters
k2 = 0.00;


% putting a copy of the first image to the end
if loop
    imgs(:, :, :, end + 1) = imgs(:, :, :, 1);
end

% cylindrical warping
nImgs = size(imgs, 4);
cylImgs = zeros(size(imgs), 'like', imgs);
for i = 1 : nImgs
    cylImgs(:, :, :, i) = cylProjection(imgs(:, :, :, i), f, k1, k2);
end


% pairwise transformation estimation
translations = estimateTranslations(cylImgs);

% transformation accumulation
accTranslations = zeros(size(translations));
accTranslations(:, :, 1) = translations(:, :, 1);
for i = 2 : nImgs
    accTranslations(:, :, i) = accTranslations(:, :, i - 1) * translations(:, :, i);
end

% new size computation & transformation refinement
width = size(cylImgs, 2);
height = size(cylImgs, 1);
if loop
    driftSlope = accTranslations(1, 3, end) / accTranslations(2, 3, end);
    newWidth = abs(round(accTranslations(2, 3, end))) + width;
    newHeight = height;
    if accTranslations(2, 3, end) < 0
        accTranslations(2, 3, :) = accTranslations(2, 3, :) - accTranslations(2, 3, end);
        accTranslations(1, 3, :) = accTranslations(1, 3, :) - accTranslations(1, 3, end);
    end
    driftMatrix = [1 -driftSlope driftSlope; 0 1 0; 0 0 1];
    for i = 1 : nImgs
        accTranslations(:, :, i) = driftMatrix * accTranslations(:, :, i);
    end
else
    maxX = width;
    minX = 1;
    maxY = height;
    minY = 1;
    frame = [[1; 1; 1], [height; 1; 1], [1; width; 1], [height; width; 1]];
    for i = 2 : nImgs 
        newFrame = accTranslations(:, :, i) * frame;
        newFrame(:, 1) = newFrame(:, 1) ./ newFrame(3, 1);
        newFrame(:, 2) = newFrame(:, 2) ./ newFrame(3, 2);
        newFrame(:, 3) = newFrame(:, 3) ./ newFrame(3, 3);
        newFrame(:, 4) = newFrame(:, 4) ./ newFrame(3, 4);
        maxX = max(maxX, max(newFrame(2, :)));
        minX = min(minX, min(newFrame(2, :)));
        maxY = max(maxY, max(newFrame(1, :)));
        minY = min(minY, min(newFrame(1, :)));
    end
    newWidth = ceil(maxX) - floor(minX) + 1;
    newHeight = ceil(maxY) - floor(minY) + 1;
    offsetX = 1 - floor(minX);
    offsetY = 1 - floor(minY);
    accTranslations(2, 3, :) = accTranslations(2, 3, :) + offsetX;
    accTranslations(1, 3, :) = accTranslations(1, 3, :) + offsetY;
end

% image mask - 1 for image & 0 for border
mask = ones(height, width);
mask = logical(cylProjection(mask, f, k1, k2));

% merging images
newImg = mergeBlendImgs(cylImgs, mask, accTranslations, newHeight, newWidth);


% cropping image
if loop
    newImg = newImg(:, width / 2 : newWidth - width / 2, :);
end

%}


