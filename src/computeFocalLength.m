function [f0, f1] = computeFocalLength(H_3X3, type)

% COMPUTEFOCALLENGTH computes focal length used to 
%     to take images
%     f = computeFocalLength(H, 'fixed') returns one focal length
%     f = computeFocalLength(H, 'variable') returns two focal lengths

if H_3x3(2,3) ~= H_3x3(1,3)
    f0 = sqrt(((H_3x3(2,3)^2) - (H_3x3(1,3)^2)) / (H_3x3(1,1)^2) + (H_3x3(1,2)^2) - (H_3x3(2,1)^2) - (H_3x3(2,2)^2));
    %%%%%%CALCULATE f1 here
end
   
if type == 'fixed'
    %%%%%%take geometric mean
end

end