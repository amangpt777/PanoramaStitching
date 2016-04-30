function stitched_img = stitchImg(img_stack, focal_length_type)

%fprintf('Number of arguments: %d\n',nargin);
% First Image should be the leftmost image
% Destination Image
nImgs = size(img_stack,4);
center_img = img_stack(:,:,:,1);

% Use RANSAC to reject outliers
% n approximated according to formulae given in slides for p and alpha
ransac_n = 50; % Max number of iteractions
ransac_eps = 2; % Acceptable alignment error % 3 pixel far

for i = 2 : nImgs
    source_img = img_stack(:,:,:,i);
    [xs, xd] = genSIFTMatches(source_img, center_img);
    center_img = [center_img zeros(size(source_img))];
    center_img(isnan(center_img)) = 0;
    while (1)
        [inliers_id, H_3x3] = runRANSAC_H(xs, xd, ransac_n, ransac_eps);
        [f0, f1] = computeFocalLength(source_img, H_3x3, focal_length_type);
        if isreal(f0) && isreal(f1)
            break
        end
    end
    %f0 = 650;
    %f1 = 28000;
     
    [inliers_id, R_3x3] = runRANSAC_R(xs, xd, ransac_n, ransac_eps, f0, f1);
    
    imgd = im2double(center_img);
    imgs = im2double(source_img);
    maskSource = logical(imgd(:,:,1));
    dest_canvas_width_height = [size(imgd, 2) size(imgd, 1)];
    [mask, dest_img] = backwardWarpImg(imgs, inv(R_3x3), dest_canvas_width_height, f0, f1);
    blended_result = blendImagePair(dest_img, mask, imgd, maskSource,...
    'blend');
    %imshow(blended_result);
    center_img = blended_result;
    %}
end
stitched_img = center_img;
end