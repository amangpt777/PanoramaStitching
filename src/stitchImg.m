function stitched_img = stitchImg(varargin)

fprintf('Number of arguments: %d\n',nargin);
% First Image should be the leftmost image
% Destination Image
center_img = varargin{1};
focal_length_type = varargin{nargin}
% Use RANSAC to reject outliers
% n approximated according to formulae given in slides for p and alpha
ransac_n = 60; % Max number of iteractions
ransac_eps = 3; % Acceptable alignment error % 3 pixel far

for i = 2 : (nargin - 1)
    source_img = varargin{i};
    [xs, xd] = genSIFTMatches(source_img, center_img);
    center_img = [center_img zeros(size(source_img))];
    center_img(isnan(center_img)) = 0;
    [inliers_id, H_3x3] = runRANSAC(xs, xd, ransac_n, ransac_eps);
    [f0, f1] = computeFocalLength(H_3x3, focal_length_type);
    R_3x3 = computeRotationalMatrix(xs, xd, f0, f1);
    
    imgd = im2double(center_img);
    imgs = im2double(source_img);
    maskSource = logical(imgd(:,:,1));
    dest_canvas_width_height = [size(imgd, 2) size(imgd, 1)];
    [mask, dest_img] = backwardWarpImg(imgs, inv(R_3x3), dest_canvas_width_height, f0, f1);
    blended_result = blendImagePair(dest_img, mask, imgd, maskSource,...
    'blend');
    imshow(blended_result);
    center_img = blended_result;
end
stitched_img = center_img;
end