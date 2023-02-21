function D = norm2mat(X1,X2)
    D = - 2 * (X1' * X2);
    D = bsxfun(@plus, D, sum(X1.^2,1)');
    D = bsxfun(@plus, D, sum(X2.^2,1));
end