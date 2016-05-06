function neighbor_imgs_raw = findNeighbor(nn_index, feature_index, m)
n_imgs = max(feature_index);
m = min(m, n_imgs-1);
neighbor_imgs_raw = zeros(n_imgs, m);
bins = 1:n_imgs;

for i = 1:n_imgs
    map = nn_index(:, (feature_index == i));
    map = map(:);
    map = feature_index(map);
    [~, sorted_index] = sort(histc(map, 1:n_imgs), 'descend');
    r = bins(sorted_index);
    r = r(r ~= i);
    neighbor_imgs_raw(i, :) = r(1:m);
end
    
