function cylProj()
upload_dir = './upload1';
loop = true;
tic
img_stack = loadImageStack(upload_dir);
stitched_img = createCylPanorama(img_stack, loop);
%figure, imshow(stitched_img);
imwrite(stitched_img, './Output1/cylindrical_panorama.png');
toc
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

function [ cylImg ] = cylProjection( img, f, k1, k2 )
img = im2double(img);
% image information
height = size(img, 1);
width = size(img, 2);
yc = (1 + height) / 2;
xc = (1 + width) / 2;

% image warping
cylImg = zeros(size(img), 'like', img);
h = 1:height; w = 1:width;
[H, W] = ndgrid(h,w);  % H: mxn, W: mxn
HH = reshape(H,1,[]); WW = reshape(W,1,[]); % column-wise
XD = (WW - xc) / f ;
YD = (HH - yc) / f ;
XYD = [XD ; YD];
rSqr = sum(XYD .^ 2, 1);
coeff = 1 + k1 * rSqr + k2 * (rSqr .^ 2);
XN = XD .* coeff;
YN = YD .* coeff;
XCap = sin(XN);
YCap = YN;
ZCap = cos(XN);
X = f * XCap ./ ZCap + xc;
Y = f * YCap ./ ZCap + yc;
x = floor(reshape(X, height, width)) ; % column segment m x n
y = floor(reshape(Y, height, width)) ; % column segment m x n
% interp2(X,Y,V,Xq,Yq), where X= 1:n, Y=1:m, [m,n] = size(V)
% X is column (width), Y is row (height)
if ndims(img) == 3
    cylImg(:, :, 1) = interp2(img(:, :, 1), x, y, 'linear');
    cylImg(:, :, 2) = interp2(img(:, :, 2), x, y, 'linear');
    cylImg(:, :, 3) = interp2(img(:, :, 3), x, y, 'linear');
else
    cylImg(:, :) = interp2(img(:, :), x, y, 'linear');
end

nan_index = find(isnan(cylImg));
if (~isempty(nan_index))
    cylImg(nan_index) = 0;
end

cylImg = im2uint8(cylImg);

end

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


%  input:   P - probability of having at least 1 success (0.99 is a good setting)
%           p - probability of a real inlier (can be pesimistic, try 0.5)
%           n - number of samples each run
%           data - the data points of interest
%           epsilon - threshold for inlier
%           settingFunctionHandle - handle to function to compute parameter values (homography, translation, etc.)
%           SSDFunctionHanle - handle to function to compute the error measure
%  output:  H - homography matrix that transforms points in image2
%               to points in image1, 3 x 3 matrix
function [bestSetting] = RANSAC(P, p, n, data, epsilon, settingFunctionHandle, SSDFunctionHandle)

k = ceil(log(1 - P) / log(1 - p^n)); % calculate number of loops
numPoints = size(data, 1); %data size
bestNumInliers = 0;
bestSet = [];

for i = 1:k
    set = 1:numPoints; % create set of all possible points
    sampleIndicies = randperm(numPoints, n);
    %set(sampleIndicies) = []; % remove samples from set
    
    samples = data(sampleIndicies,:,:);
    setting = settingFunctionHandle(samples(:,:,1), samples(:,:,2)); % get current settings
    
    % loop over the rest to find inliers
    remaining = set;
    numInliers = 0;
    for j = remaining
        SSD = SSDFunctionHandle(data(j,:,:), setting);
        
        if SSD < epsilon
            numInliers = numInliers + 1;
        else % not inlier, remove from set
            set(set == j) = [];
        end
    end
    
    % check if new best
    if numInliers > bestNumInliers
        bestSet = set;
        bestNumInliers = numInliers;
    end
end

% compute setting for best set
bestData = data(bestSet,:,:);
bestSetting = settingFunctionHandle(bestData(:,:,1), bestData(:,:,2)); % get best settings using all inliers

end
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
end