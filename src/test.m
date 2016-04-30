
height = 4;
width = 6;
f = 2;
k1 = 2;
k2 = 2;
yc = (1 + height) / 2;
xc = (1 + width) / 2;

h = 1:height; w = 1:width;
[H, W] = ndgrid(h,w);  % H: mxn, W: mxn
HH = reshape(H,1,[]); WW = reshape(W,1,[]); % column-wise
XD = (WW - xc) / f ;
YD = (HH - yc) / f ;
XYD = [XD ; YD];
rSqr = sum(XYD .^2 , 1);
coeff = 1 + k1 * rSqr + k2 * (rSqr .^ 2);
XN = XD .* coeff;
YN = YD .* coeff;
XCap = sin(XN);
YCap = YN;
ZCap = cos(XN);
X = f * XCap ./ ZCap + xc;
Y = f * YCap ./ ZCap + yc;
x = floor(reshape(X, height, width)) ; % column segment m x n
y = floor(reshape(Y, height, width)) ; % column segment m x n
