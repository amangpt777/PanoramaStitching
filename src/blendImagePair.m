function out_img = blendImagePair(wrapped_imgs, masks, wrapped_imgd, maskd, mode)
if strcmp(mode,'overlay')
    lmaskd = logical(maskd);  % 0 and 1 mask to add imgd
    rmaskd = ~maskd;  % reverse maskd to add imgs
    
    img1 = im2double(wrapped_imgs) .* cat(3, rmaskd, rmaskd, rmaskd);
    nan_index = find(isnan(img1));
    if (~isempty(nan_index))
        img1(nan_index) = 0;
    end
    
    img2 = im2double(wrapped_imgd) .* cat(3, lmaskd, lmaskd, lmaskd);
    nan_index = find(isnan(img2));
    if (~isempty(nan_index))
        img2(nan_index) = 0;
    end

    out_img = img1 + img2;
    
elseif strcmp(mode,'blend')
    imasks = uint8(logical(masks)); % 0 and 1 mask
    imaskd = uint8(logical(maskd)); % 0 and 1 mask
    
    wmasks = imasks;
    wmasks(imaskd == 1) = 0;
    wmasks = bwdist(wmasks); % 
    %figure, imshow(mat2gray(wmasks));
    %convert distance to weight
    wmasks(wmasks == 0) = 1;  % distance equals to zero, set weight to 1
    wmasks = wmasks .^-1;
    %figure, imshow(mat2gray(wmasks));
    
    wmaskd = imaskd;
    wmaskd(imasks == 1) = 0;
    wmaskd = bwdist(wmaskd);
    %figure, imshow(mat2gray(wmaskd));
    % convert distance to weight
    wmaskd(wmaskd == 0) = 1;  % distance equals to zero, sign weight to 1
    wmaskd = wmaskd .^-1;
    %figure, imshow(mat2gray(wmaskd));
    
    % calculate weighted weight.
    wmasksum = wmasks + wmaskd;
    wmasks = wmasks ./ wmasksum;
    wmaskd = wmaskd ./ wmasksum;
    
    % if there is any nan value, will cause the picture is black.
    img1 = im2double(wrapped_imgs) .* cat(3, wmasks, wmasks, wmasks);
    nan_index = find(isnan(img1));
    if (~isempty(nan_index))
        img1(nan_index) = 0;
    end
    
    img2 = im2double(wrapped_imgd) .* cat(3, wmaskd, wmaskd, wmaskd);
    nan_index = find(isnan(img2));
    if (~isempty(nan_index))
        img2(nan_index) = 0;
    end
    
    out_img = img1 + img2;
else
    error('Only support overlay mode and blend mode');
end
