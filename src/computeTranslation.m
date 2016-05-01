function T = computeTranslation( cp1, cp2 )
% cp1: np x 2
% cp2: np x 2
% T: 3 x 3
n = size(cp1,1);
A = [ones(n,1) zeros(n,1); zeros(n,1) ones(n,1)];
b = [cp1(:,1) - cp2(:,1); cp1(:,2) - cp2(:,2)];
t = A \ b;
T = [1 0 t(1); 0 1 t(2); 0 0 1];

end

