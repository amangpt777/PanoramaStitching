function runStitch(varargin)
%
% Usage:
% runStitch                     : list all the registered functions
% runStitch('function_name')      : execute a specific test

% Settings to make sure images are displayed without borders
orig_imsetting = iptgetpref('ImshowBorder');
iptsetpref('ImshowBorder', 'tight');
temp1 = onCleanup(@()iptsetpref('ImshowBorder', orig_imsetting));

fun_handles = {@challengeStitch, ...
               @cylProj};

% Call test harness
runTests(varargin, fun_handles);


%%
function challengeStitch()
% Test image stitching

focal_length_type = 'variable';

upload_dir = 'upload';
tic
img_stack = loadImageStack(upload_dir);
%img_stack = autoDetect(upload_dir);
stitched_img = stitchImg(img_stack, focal_length_type);
%figure, imshow(stitched_img);
imwrite(stitched_img, 'mountain_panorama.png');
toc
%}

%%
function cylProj()
upload_dir = 'upload1';
loop = true;
tic
%img_stack = loadImageStack(upload_dir);
img_stack = autoDetect(upload_dir);
stitched_img = createCylPanorama(img_stack, loop);
%figure, imshow(stitched_img);
imwrite(stitched_img, 'EH_panorama.png');
toc
%}