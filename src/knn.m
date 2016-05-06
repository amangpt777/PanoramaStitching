function [nn_index, feature_index] = knn(F)
n_imgs = length(F);
F_t = cell2mat(F);
n_index = size(F_t, 1);
feature_index = zeros(1, n_index);

n0 = 1;
for i = 1:n_imgs
    n1 = n0 + size(F{i}, 1) - 1;
    feature_index(n0:n1) = i;
    n0 = n1 + 1;
end

n_nn = 4;
if n_imgs < 4
    n_nn = n_imgs;
end

MdlKDT = KDTreeSearcher(F_t);
nn_index = knnsearch(MdlKDT, F_t, 'K', n_nn);
nn_index = nn_index';
nn_index = nn_index(2:n_nn, :);
