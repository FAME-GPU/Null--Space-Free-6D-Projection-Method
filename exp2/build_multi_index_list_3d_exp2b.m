function alphaList = build_multi_index_list_3d_exp2b(p)
%BUILD_MULTI_INDEX_LIST_3D_EXP2B All 3-D multi-indices with total degree <= p.
    p = round(p);
    if p < 0
        error('build_multi_index_list_3d_exp2b:BadOrder','Taylor order must be nonnegative.');
    end
    L = nchoosek(p+3,3);
    alphaList = zeros(L,3);
    c = 0;
    for a1 = 0:p
        for a2 = 0:(p-a1)
            for a3 = 0:(p-a1-a2)
                c = c + 1;
                alphaList(c,:) = [a1 a2 a3];
            end
        end
    end
    if c ~= L
        error('build_multi_index_list_3d_exp2b:CountBug','Unexpected number of Taylor terms.');
    end
end
