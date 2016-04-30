function challengeStitch()
% Test image stitching

focal_length_type = 'variable';
upload_dir = './upload';
img_stack = loadImageStack(upload_dir);
% Should pass the leftMost image first

stitched_img = stitchImg(img_stack, focal_length_type);
%figure, imshow(stitched_img);
imwrite(stitched_img, './Output/mountain_panorama.png');
%}
end

function [rgb_stack] = loadImageStack(upload_dir)

%fnames = dir([upload_dir '/*.jpg']);
fnames = dir([upload_dir]);
names = {fnames.name};
% remove . and .. files
names = names(cellfun(@(x) x(1), names) ~= '.');

cac = regexprep( names,                          ...
                 '^([A-z]+)(\d+)(\.\w+)$', '$2' );
[ ~, ixs ] = sort(str2double(cac));
names = names(ixs);

img = imread([upload_dir '/' char(names(1))]);
[m,n,~] = size(img);
k = length(names);
rgb_stack = zeros(m,n,3,k, 'uint8'); % must define type as uint8
for i = 1: k
    img = imread([upload_dir '/' char(names(i))]);
    rgb_stack(:,:,:,i) = img;
end

end


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
%im = imread('vase1.png');
for i = 2 : nImgs
    source_img = img_stack(:,:,:,i);
%    imwrite(source_img, './Output/source_img.png');
    [xs, xd] = genSIFTMatches(source_img, center_img);
%    imwrite(source_img, './Output/source_img_1.png');
    center_img = [center_img zeros(size(source_img))];
    center_img(isnan(center_img)) = 0;
    while (1)
        [inliers_id, H_3x3] = runRANSAC_H(xs, xd, ransac_n, ransac_eps);
        [f0, f1] = computeFocalLength(source_img, H_3x3, focal_length_type);
        if isreal(f0) && isreal(f1)
            break
        end
    end
%    imwrite(im, './Output/vase1_3.png');
    %f0 = 650;
    %f1 = 28000;
     
    [inliers_id, R_3x3] = runRANSAC_R(xs, xd, ransac_n, ransac_eps, f0, f1);
%    imwrite(im, './Output/vase1_4.png');
    imgd = im2double(center_img);
    imgs = im2double(source_img);
    maskSource = logical(imgd(:,:,1));
    dest_canvas_width_height = [size(imgd, 2) size(imgd, 1)];
    [mask, dest_img] = backwardWarpImg(imgs, inv(R_3x3), dest_canvas_width_height, f0, f1);
    blended_result = blendImagePair(dest_img, mask, imgd, maskSource,...
    'blend');
    imshow(blended_result);
    center_img = blended_result;
    %}
end
%imwrite(im, './Output/vase1_5.png');
stitched_img = center_img;
end

function [xs, xd] = genSIFTMatches(imgs, imgd)

% Add the sift library to path if the sift mex file cannot be found
% Here we assume the sift_lib is placed at a predefined location 
% (relative path)
if ~isequal(exist('vl_sift'), 3)
    sift_lib_dir = fullfile('sift_lib', ['mex' lower(computer)]);
    orig_path = addpath(sift_lib_dir);
    % Restore the original path upon function completion 
    temp = onCleanup(@()path(orig_path));
end

imgs = im2single(imgs); gray_s = rgb2gray(imgs);
imgd = im2single(imgd); gray_d = rgb2gray(imgd);

[Fs, Ds] = vl_sift(gray_s);
% Each column of Fs is a feature frame and has the format [X; Y; S; TH],
% where X, Y is the (fractional) center of the frame, S is the scale and TH
% is the orientation (in radians)
% Ds is the descriptor of the corresponding frame in F.
[Fd, Dd] = vl_sift(gray_d);

[matches, scores] = vl_ubcmatch(Ds, Dd);
% matches: 2xn matrix, scores: 1xn matrix
% The two rows of matches store the indices of Ds and Dd that match with
% each other

xs = Fs(1:2, matches(1, :))';
xd = Fd(1:2, matches(2, :))';
% xs and xd are the centers of matched frames
% xs and xd are nx2 matrices
end

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
end

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
end

function [f0, f1] = computeFocalLength(img, H_3x3, type)

% COMPUTEFOCALLENGTH computes focal length used to 
%     to take images
%     f = computeFocalLength(H, 'fixed') returns one focal length
%     f = computeFocalLength(H, 'variable') returns two focal lengths


if H_3x3(1,1) * H_3x3(1,3) - H_3x3(1,2) * H_3x3(2,2) ~= 0
    numer = H_3x3(1,3) * H_3x3(2,3);
    denom = H_3x3(1,1) * H_3x3(2,1) + H_3x3(1,2) * H_3x3(2,2);
    f0 = sqrt( -numer / denom );
end

if H_3x3(3,1) * H_3x3(3,2) ~= 0
    numer = H_3x3(1,1) * H_3x3(1,2) + H_3x3(2,1) * H_3x3(2,2);
    denom = H_3x3(3,1) * H_3x3(3,2);
    f1 = sqrt( - numer / denom );

end
if strcmp(type, 'fixed')
    %%%%%%take geometric mean
    f0 = sqrt(f0 * f1);
    f1 = f0;
end


%}

%{
if ~exist('nx', 'var')
    [ny, nx] = size(img);
end
c_init = [nx;ny]/2 - 0.5; % initialize at the center of the image
k_init = [0;0;0;0;0]; % initialize to zero (no distortion)

% Compute explicitely the focal length using all the (mutually orthogonal) vanishing points
% note: The vanihing points are hidden in the planar collineations H_kk

A = [];
b = [];

% matrix that subtract the principal point:
Sub_cc = [1 0 -c_init(1);0 1 -c_init(2);0 0 1];
H_3x3 = Sub_cc * H_3x3;

% Extract vanishing points (direct and diagonals):
        
        V_hori_pix = H_3x3(:,1);
        V_vert_pix = H_3x3(:,2);
        V_diag1_pix = (H_3x3(:,1)+H_3x3(:,2))/2;
        V_diag2_pix = (H_3x3(:,1)-H_3x3(:,2))/2;
        
        V_hori_pix = V_hori_pix/norm(V_hori_pix);
        V_vert_pix = V_vert_pix/norm(V_vert_pix);
        V_diag1_pix = V_diag1_pix/norm(V_diag1_pix);
        V_diag2_pix = V_diag2_pix/norm(V_diag2_pix);
        
        a1 = V_hori_pix(1);
        b1 = V_hori_pix(2);
        c1 = V_hori_pix(3);
        
        a2 = V_vert_pix(1);
        b2 = V_vert_pix(2);
        c2 = V_vert_pix(3);
        
        a3 = V_diag1_pix(1);
        b3 = V_diag1_pix(2);
        c3 = V_diag1_pix(3);
        
        a4 = V_diag2_pix(1);
        b4 = V_diag2_pix(2);
        c4 = V_diag2_pix(3);
        
        A_kk = [a1*a2  b1*b2;
            a3*a4  b3*b4];
        
        b_kk = -[c1*c2;c3*c4];
        
        
        A = [A;A_kk];
        b = [b;b_kk];

        % use all the vanishing points to estimate focal length:


% Select the model for the focal. (solution to Gerd's problem)
two_focals_init = 0;
if ~two_focals_init
    if b'*(sum(A')') < 0,
        two_focals_init = 1;
    end;
end;

    

if two_focals_init
    % Use a two focals estimate:
    f_init = sqrt(abs(1./(inv(A'*A)*A'*b))); % if using a two-focal model for initial guess
else
    % Use a single focal estimate:
    f_init = sqrt(b'*(sum(A')') / (b'*b)) * ones(2,1); % if single focal length model is used
end;


est_aspect_ratio = 0;
if ~est_aspect_ratio,
    f_init(1) = (f_init(1)+f_init(2))/2;
    f_init(2) = f_init(1);
end;

alpha_init = 0;

%f_init = sqrt(b'*(sum(A')') / (b'*b)) * ones(2,1); % if single focal length model is used


% Global calibration matrix (initial guess):

KK = [f_init(1) alpha_init*f_init(1) c_init(1);0 f_init(2) c_init(2); 0 0 1];
inv_KK = inv(KK);


cc = c_init;
fc = f_init;
kc = k_init;
alpha_c = alpha_init;

f0 = f_init(1);
f1 = f_init(2);


fprintf(1,'\n\nCalibration parameters after initialization:\n\n');
fprintf(1,'Focal Length:          fc = [ %3.5f   %3.5f ]\n',fc);
fprintf(1,'Principal point:       cc = [ %3.5f   %3.5f ]\n',cc);
fprintf(1,'Skew:             alpha_c = [ %3.5f ]   => angle of pixel = %3.5f degrees\n',alpha_c,90 - atan(alpha_c)*180/pi);
fprintf(1,'Distortion:            kc = [ %3.5f   %3.5f   %3.5f   %3.5f   %5.5f ]\n',kc);
%}
end

function [mask, result_img] = backwardWarpImg(src_img, resultToSrc_H,...
    dest_canvas_width_height, f0, f1)

dest_canvas_width = dest_canvas_width_height(1,1);
dest_canvas_height = dest_canvas_width_height(1,2);

h = 1:dest_canvas_height; w = 1:dest_canvas_width;
% pay attentation to the difference between meshgrid and ndgrid
% [m,n] = size(V)
%meshgrid : row-wise n x m 
%ndgrid: column-wise m x n
[H, W] = ndgrid(h,w);  % H: mxn, W: mxn
HH = reshape(H,1,[]); WW = reshape(W,1,[]); % column-wise
HWZ = [WW; HH; ones(1, dest_canvas_height * dest_canvas_width) * f1];

% inv(H) * [xd, yd, 1]' = [xs, ys, zs]'
S =  resultToSrc_H * HWZ;
SHW = [S(1,:)./S(3,:).* f0; S(2,:)./S(3,:).*f0]; % first row is width, column-segment


x = reshape(SHW(1, :), dest_canvas_width_height(2), ...
    dest_canvas_width_height(1)) ; % column segment m x n
y = reshape(SHW(2, :), dest_canvas_width_height(2), ...
    dest_canvas_width_height(1));
result_img = zeros(dest_canvas_width_height(2), dest_canvas_width_height(1));

% interp2(X,Y,V,Xq,Yq), where X= 1:n, Y=1:m, [m,n] = size(V)
% X is column (width), Y is row (height)
result_img(:, :, 1) = interp2(src_img(:, :, 1), x, y, 'linear');
result_img(:, :, 2) = interp2(src_img(:, :, 2), x, y, 'linear');
result_img(:, :, 3) = interp2(src_img(:, :, 3), x, y, 'linear');

nan_index = find(isnan(result_img));
if (~isempty(nan_index))
    result_img(nan_index) = 0;
end
mask = logical(result_img(:, :, 1));
end


function out_img = blendImagePair(wrapped_imgs, masks, wrapped_imgd, maskd, mode)
if strcmp(mode,'overlay')
    lmaskd = logical(maskd);  % 0 and 1 mask to add imgd
    rmaskd = ~maskd;  % reverse maskd to add imgs
    
    img1 = im2double(wrapped_imgs) .* cat(3, rmaskd, rmaskd, rmaskd);
    nan_index = find(isnan(img1));
    if (~isempty(nan_index))
        img1(nan_index) = 0;
    end
    
    img2 = im2double(wrapped_imgd) .* cat(3, lmaskd, lmaskd, lmaskd);
    nan_index = find(isnan(img2));
    if (~isempty(nan_index))
        img2(nan_index) = 0;
    end

    out_img = img1 + img2;
    
elseif strcmp(mode,'blend')
    imasks = uint8(logical(masks)); % 0 and 1 mask
    imaskd = uint8(logical(maskd)); % 0 and 1 mask
    
    wmasks = imasks;
    wmasks(imaskd == 1) = 0;
    wmasks = bwdist(wmasks); % 
    %figure, imshow(mat2gray(wmasks));
    %convert distance to weight
    wmasks(wmasks == 0) = 1;  % distance equals to zero, set weight to 1
    wmasks = wmasks .^-1;
    %figure, imshow(mat2gray(wmasks));
    
    wmaskd = imaskd;
    wmaskd(imasks == 1) = 0;
    wmaskd = bwdist(wmaskd);
    %figure, imshow(mat2gray(wmaskd));
    % convert distance to weight
    wmaskd(wmaskd == 0) = 1;  % distance equals to zero, sign weight to 1
    wmaskd = wmaskd .^-1;
    %figure, imshow(mat2gray(wmaskd));
    
    % calculate weighted weight.
    wmasksum = wmasks + wmaskd;
    wmasks = wmasks ./ wmasksum;
    wmaskd = wmaskd ./ wmasksum;
    
    % if there is any nan value, will cause the picture is black.
    img1 = im2double(wrapped_imgs) .* cat(3, wmasks, wmasks, wmasks);
    nan_index = find(isnan(img1));
    if (~isempty(nan_index))
        img1(nan_index) = 0;
    end
    
    img2 = im2double(wrapped_imgd) .* cat(3, wmaskd, wmaskd, wmaskd);
    nan_index = find(isnan(img2));
    if (~isempty(nan_index))
        img2(nan_index) = 0;
    end
    
    out_img = img1 + img2;
else
    error('Only support overlay mode and blend mode');
end
end

function dest_pts_nx2 = applyHomography(H_3x3, src_pts_nx2)
xs = src_pts_nx2(:,1);
ys = src_pts_nx2(:,2);
n = numel(xs);

column1 = ones(n,1);
A = [xs ys column1] * H_3x3';
B = [A(:,1)./A(:,3) A(:,2)./A(:,3)];

dest_pts_nx2 = B;
end

function dest_pts_nx2 = applyRotationalMatrix(R_3x3, src_pts_nx2, f0, f1)
xs = src_pts_nx2(:,1);
ys = src_pts_nx2(:,2);
n = numel(xs);

column1 = ones(n,1) * f0;
A = [xs ys column1] * R_3x3';
B = [A(:,1)./A(:,3).* f1 A(:,2)./A(:,3).* f1];

dest_pts_nx2 = B;
end

function H_3x3 = computeHomography(src_pts_nx2, dest_pts_nx2)
xs = src_pts_nx2(:,1);
ys = src_pts_nx2(:,2);
xd = dest_pts_nx2(:,1);
yd = dest_pts_nx2(:,2);

n = numel(xs);

column1 = ones(n,1);
column0 = zeros(n,3);

Ax = [xs ys column1 column0 -xd.*xs -xd.*ys -xd];
Ay = [column0 xs ys column1 -yd.*xs -yd.*ys -yd];

A = [Ax;Ay];

[V,D] = eig(A'*A); % V is the matrix whose column are the corresponding vectors
% the return eigenvalues is not in sorted order, we must sort it
ev = diag(D);
[ev_sorted, ind] = sort(ev);
% l = norm(V(:,1));  % used to check the ||h|| = 1
H_3x3 = reshape(V(:,ind(1)), [3,3])'; % reshape is column-wise, we want row-wise
                                 % we need to transpose it

%{
% Solve equations using SVD
X = xs'; Y = ys';
x = xd'; y = yd';
rows0 = zeros(3, n);
rowsXY = -[X; Y; ones(1,n)];
hx = [rowsXY; rows0; x.*X; x.*Y; x];
hy = [rows0; rowsXY; y.*X; y.*Y; y];
h = [hx hy]; % 9x2n
if n == 4
    [U, ~, ~] = svd(h);
else
    [U, ~, ~] = svd(h, 'econ');
end
v = reshape(U(:,9), 3, 3)'
%}
end
