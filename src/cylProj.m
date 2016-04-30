%% warp image into cylindrical coordinates and correct radial distortion
%  input:   img - source image
%           f - focal length
%           k1, k2 - radial distortion parameters
%  output:  cylImg - cylindrical warpped images
function [ cylImg ] = cylProj( img, f, k1, k2 )
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



