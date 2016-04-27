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









