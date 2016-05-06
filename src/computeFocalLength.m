function [f0, f1] = computeFocalLength(img, H_3x3, type)

% COMPUTEFOCALLENGTH computes focal length used to 
%     to take images
%     f = computeFocalLength(H, 'fixed') returns one focal length
%     f = computeFocalLength(H, 'variable') returns two focal lengths

%{
if H_3x3(1,1) * H_3x3(1,3) - H_3x3(1,2) * H_3x3(2,2) ~= 0
    numer = H_3x3(1,3) * H_3x3(2,3);
    denom = H_3x3(1,1) * H_3x3(2,1) + H_3x3(1,2) * H_3x3(2,2);
    f0 = sqrt( -numer / denom );
end

if H_3x3(3,1) * H_3x3(3,2) ~= 0
    numer = H_3x3(1,1) * H_3x3(1,2) + H_3x3(2,1) * H_3x3(2,2);
    denom = H_3x3(3,1) * H_3x3(3,2);
    f1 = sqrt( - numer / denom );

end
if strcmp(type, 'fixed')
    %%%%%%take geometric mean
    f0 = sqrt(f0 * f1);
    f1 = f0;
end


%}


if ~exist('nx', 'var')
    [ny, nx] = size(img);
end
c_init = [nx;ny]/2 - 0.5; % initialize at the center of the image
k_init = [0;0;0;0;0]; % initialize to zero (no distortion)

% Compute explicitely the focal length using all the (mutually orthogonal) vanishing points
% note: The vanihing points are hidden in the planar collineations H_kk

A = [];
b = [];

% matrix that subtract the principal point:
Sub_cc = [1 0 -c_init(1);0 1 -c_init(2);0 0 1];
H_3x3 = Sub_cc * H_3x3;

% Extract vanishing points (direct and diagonals):
        
        V_hori_pix = H_3x3(:,1);
        V_vert_pix = H_3x3(:,2);
        V_diag1_pix = (H_3x3(:,1)+H_3x3(:,2))/2;
        V_diag2_pix = (H_3x3(:,1)-H_3x3(:,2))/2;
        
        V_hori_pix = V_hori_pix/norm(V_hori_pix);
        V_vert_pix = V_vert_pix/norm(V_vert_pix);
        V_diag1_pix = V_diag1_pix/norm(V_diag1_pix);
        V_diag2_pix = V_diag2_pix/norm(V_diag2_pix);
        
        a1 = V_hori_pix(1);
        b1 = V_hori_pix(2);
        c1 = V_hori_pix(3);
        
        a2 = V_vert_pix(1);
        b2 = V_vert_pix(2);
        c2 = V_vert_pix(3);
        
        a3 = V_diag1_pix(1);
        b3 = V_diag1_pix(2);
        c3 = V_diag1_pix(3);
        
        a4 = V_diag2_pix(1);
        b4 = V_diag2_pix(2);
        c4 = V_diag2_pix(3);
        
        A_kk = [a1*a2  b1*b2;
            a3*a4  b3*b4];
        
        b_kk = -[c1*c2;c3*c4];
        
        
        A = [A;A_kk];
        b = [b;b_kk];

        % use all the vanishing points to estimate focal length:


% Select the model for the focal. (solution to Gerd's problem)
two_focals_init = 0;
if ~two_focals_init
    if b'*(sum(A')') < 0,
        two_focals_init = 1;
    end;
end;

    

if two_focals_init
    % Use a two focals estimate:
    f_init = sqrt(abs(1./(inv(A'*A)*A'*b))); % if using a two-focal model for initial guess
else
    % Use a single focal estimate:
    f_init = sqrt(b'*(sum(A')') / (b'*b)) * ones(2,1); % if single focal length model is used
end;


est_aspect_ratio = 0;
if ~est_aspect_ratio,
    f_init(1) = (f_init(1)+f_init(2))/2;
    f_init(2) = f_init(1);
end;

alpha_init = 0;

%f_init = sqrt(b'*(sum(A')') / (b'*b)) * ones(2,1); % if single focal length model is used


% Global calibration matrix (initial guess):

KK = [f_init(1) alpha_init*f_init(1) c_init(1);0 f_init(2) c_init(2); 0 0 1];
inv_KK = inv(KK);


cc = c_init;
fc = f_init;
kc = k_init;
alpha_c = alpha_init;

f0 = f_init(1);
f1 = f_init(2);

%{
fprintf(1,'\n\nCalibration parameters after initialization:\n\n');
fprintf(1,'Focal Length:          fc = [ %3.5f   %3.5f ]\n',fc);
fprintf(1,'Principal point:       cc = [ %3.5f   %3.5f ]\n',cc);
fprintf(1,'Skew:             alpha_c = [ %3.5f ]   => angle of pixel = %3.5f degrees\n',alpha_c,90 - atan(alpha_c)*180/pi);
fprintf(1,'Distortion:            kc = [ %3.5f   %3.5f   %3.5f   %3.5f   %5.5f ]\n',kc);
%}