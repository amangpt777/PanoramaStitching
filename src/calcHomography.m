function [neighbor_imgs] = calcHomography(F, P, neighbor_imgs_raw)
[n_imgs, n_nn] = size(neighbor_imgs_raw);
neighbor_imgs.lists = cell(1, n_imgs);
neighbor_imgs.Hs = cell(1, n_imgs);
neighbor_imgs.inlier_original = cell(1, n_imgs);
neighbor_imgs.inlier_target = cell(1, n_imgs);

for i = 1:n_imgs
    nn_list = neighbor_imgs_raw(i, :);
    neighbor_imgs.lists{i} = zeros(1, n_nn);
    neighbor_imgs.Hs{i} = cell(1, n_nn);
    neighbor_imgs.inlier_original{i} = cell(1, n_nn);
    neighbor_imgs.inlier_target{i} = cell(1, n_nn);
    
    for j = 1:n_nn
        jj = nn_list(j);
        index_pairs = matchFeatures(F{jj}, F{i});
        
        if size(index_pairs, 1) == 0
            continue;
        end
        matched_pts_original = P{jj}(index_pairs(:,1), :);
        matched_pts_target = P{i}(index_pairs(:,2), :);
        [H, inlier_pts_original, inlier_pts_target, status] = ... 
            estimateGeometricTransform...
            (matched_pts_original, matched_pts_target, 'projective');
        
        nf = size(index_pairs, 1);
        ni = size(inlier_pts_target, 1);
        %[nf, ni, i, jj]
        if ni > 8 + 0.2*nf
        %if status == 0
            neighbor_imgs.lists{i}(j) = jj;
            neighbor_imgs.Hs{i}{j} = H;
            neighbor_imgs.inlier_original{i}{j} = double(inlier_pts_original);
            neighbor_imgs.inlier_target{i}{j} = double(inlier_pts_target);          
        end
    end
    R = neighbor_imgs.lists{i};
    neighbor_imgs.lists{i} = R(R ~= 0);
    R = neighbor_imgs.Hs{i};
    neighbor_imgs.Hs{i} = R(~cellfun('isempty',R));
    R = neighbor_imgs.inlier_original{i};
    neighbor_imgs.inlier_original{i} = R(~cellfun('isempty',R));
    R = neighbor_imgs.inlier_target{i};
    neighbor_imgs.inlier_target{i} = R(~cellfun('isempty',R));
    %%%target = original * H;
end