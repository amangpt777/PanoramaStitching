function [rgb_stack, gray_stack] = loadFocalStack(focal_stack_dir)

%fnames = dir([focal_stack_dir '/*.jpg']);
fnames = dir([focal_stack_dir]);
names = {fnames.name};
% remove . and .. files
names = names(cellfun(@(x) x(1), names) ~= '.');

cac = regexprep( names,                          ...
                 '^([A-z]+)(\d+)(\.\w+)$', '$2' );
[ ~, ixs ] = sort(str2double(cac));
names = names(ixs);

img = imread([focal_stack_dir '/' char(names(1))]);
[m,n,~] = size(img);
k = length(names);
rgb_stack = zeros(m,n,3*k);
gray_stack = zeros(m,n,k);
for i = 1: k
    img = imread([focal_stack_dir '/' char(names(i))]);
    rgb_stack(:,:,3*(i-1)+1:3*(i-1)+3) = img;
    gray_stack(:,:,i) = rgb2gray(img);
end









