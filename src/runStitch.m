function runStitch(varargin)
%
% Usage:
% runStitch                     : list all the registered functions
% runStitch('function_name')      : execute a specific test

% Settings to make sure images are displayed without borders
orig_imsetting = iptgetpref('ImshowBorder');
iptsetpref('ImshowBorder', 'tight');
temp1 = onCleanup(@()iptsetpref('ImshowBorder', orig_imsetting));

fun_handles = {@challengeStitch};

% Call test harness
runTests(varargin, fun_handles);


%%
function challengeStitch()
% Test image stitching

focal_length_type = 'variable';

% stitch three images
imgc = im2single(imread('mountain_center.png'));
imgl = im2single(imread('mountain_left.png'));
imgr = im2single(imread('mountain_right.png'));

% You are free to change the order of input arguments
% Should pass the leftMost image first
stitched_img = stitchImg(imgl, imgc, imgr, focal_length_type);
%figure, imshow(stitched_img);
imwrite(stitched_img, 'mountain_panorama.png');

%%
function challenge1f()
imgl = im2single(imread('left.png'));
imgc = im2single(imread('center.png'));
imgr = im2single(imread('right.png'));
stitched_img = stitchImg(imgl, imgc, imgr);
%figure, imshow(stitched_img);
imwrite(stitched_img, 'panorama.png');
% Your own panorama
