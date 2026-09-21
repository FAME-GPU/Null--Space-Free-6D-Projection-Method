function alphaList = build_multi_index_list_3d_exp3(p)
%BUILD_MULTI_INDEX_LIST_3D_EXP3 All alpha in N_0^3 with |alpha|<=p.
    L=nchoosek(p+3,3);
    alphaList=zeros(L,3);
    c=0;
    for a1=0:p
        for a2=0:(p-a1)
            for a3=0:(p-a1-a2)
                c=c+1;
                alphaList(c,:)=[a1 a2 a3];
            end
        end
    end
    alphaList=alphaList(1:c,:);
end
