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
SHW = [S(1,:)./S(3,:); S(2,:)./S(3,:)]; % first row is width, column-segment


x = reshape(SHW(1, :), dest_canvas_width_height(2), ...
    dest_canvas_width_height(1)) / f0 ; % column segment m x n
y = reshape(SHW(2, :), dest_canvas_width_height(2), ...
    dest_canvas_width_height(1)) / f0;
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





