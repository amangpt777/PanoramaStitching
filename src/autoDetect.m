function imgs = autoDetect(input_dir)
fnames = dir(input_dir);
names = {fnames.name};
% remove . and .. files
names = names(cellfun(@(x) x(1), names) ~= '.');
n_imgs = length(names); % Number of files found
images = cell(1, n_imgs);
%rand_index = randperm(n_imgs);
%names = names(rand_index);

for i = 1:n_imgs
   images{i} = imread([input_dir '/' char(names(i))]);
end

image_size = size(images{1});
height = image_size(1); width = image_size(2);
[F, P, F_strongest] = genFeatures(images);

[nn_index, feature_index] = knn(F_strongest);
neighbor_imgs_raw = findNeighbor(nn_index, feature_index, 6);
neighbor_imgs = calcHomography(F, P, neighbor_imgs_raw);
order = findOrder(neighbor_imgs.lists, neighbor_imgs.inlier_original, ...
    neighbor_imgs.inlier_target);
imgs = zeros([height, width, 3, n_imgs], 'like', images{1});
for i = 1:numel(order)
    imgs(:, :, :, i) = images{order(i)};
end

function order = findOrder(lists, inliers_o, inliers_t)
%%%%%%%% initialization %%%%%%%%%%%%%%%
n_imgs = length(lists);


ends = [0, 0];
j = 1;
for i = 1:n_imgs
    if (length(lists{i}) == 1)
        ends(j) = i;
        j = j + 1;
    end
end
order = zeros(1, n_imgs);
if ends(1) == 0
    order(1) = 1;
else
    min1 = min(inliers_o{ends(1)}{1}(:, 1));
    min2 = min(inliers_o{ends(2)}{1}(:, 1));
    if (min1 > min2)
        order(1) = ends(1);
    else
        order(1) = ends(2);
    end
end
for i = 2:n_imgs
    nn_list = lists{order(i-1)};    
    for j = 1:numel(nn_list)
        min1 = min(inliers_o{order(i-1)}{j}(:, 1));
        min2 = min(inliers_t{order(i-1)}{j}(:, 1));
        if (min1 > min2)
            order(i) = nn_list(j);
        end
    end
end


    
% max_matches = 0;
% for i = 1:n_cc
%     nn_list = lists{cc(i)};
%     count = 0;
%     for j = 1:numel(nn_list)
%         n = size(inliers{cc(i)}{j}, 1);
%         n_inliers(cc(i), nn_list(j)) = n;
%         count = count + n;
%     end
%     if max_matches < count
%         max_matches = count;
%         max_index = cc(i);
%     end
% end
% queue(:, 1) = [0; max_index];
% visited(max_index) = 1;
% 
% for i = 2:n_cc
%     max_matches = 0;
%     for j = total(~visited)
%         temp = n_inliers(j, :);
%         count = sum(temp(logical(visited)));
%         if max_matches < count
%             max_matches = count;
%             max_index = j;
%         end
%     end
%     queue(2, i) = max_index;
%     temp = n_inliers(max_index, :);
%     max_matches = max(temp(logical(visited)));
%     queue(1, i) = find(temp == max_matches);
%     visited(max_index) = 1;
% end
% ends = setdiff(queue(2, :), queue(1, :));
% min1 = min(inliers{ends(1)}{1}(:, 1))
% min2 = min(inliers{ends(2)}{1}(:, 1))
% order = zeros(1, n_cc);
% if (min1 > min2)
%     order(1) = ends(1);
% else
%     order(1) = ends(2);
% end
% 
% for i = 2:n_cc
%     order(i) = queue(1, find(order(i-1) == queue(2, :)));
% end

    
    