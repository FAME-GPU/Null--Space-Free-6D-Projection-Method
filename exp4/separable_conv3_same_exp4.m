function Y = separable_conv3_same_exp4(X,k)
%SEPARABLE_CONV3_SAME_EXP4 Zero-padded separable 3D convolution, same size.
    k=k(:);
    Y=convn(X,reshape(k,[],1,1),'same');
    Y=convn(Y,reshape(k,1,[],1),'same');
    Y=convn(Y,reshape(k,1,1,[]),'same');
end
