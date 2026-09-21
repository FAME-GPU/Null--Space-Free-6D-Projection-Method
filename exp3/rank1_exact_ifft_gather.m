function y = rank1_exact_ifft_gather(Yfft,plan)
%RANK1_EXACT_IFFT_GATHER Apply 1-D IFFT and gather retained target codes.

    if isvector(Yfft), Yfft=Yfft(:); end
    if size(Yfft,1)~=plan.M
        error('rank1_exact_ifft_gather:BadSize','Yfft must have plan.M rows.');
    end

    yEmbed=ifft(Yfft,[],1);
    y=yEmbed(plan.targetIdx,:);
end
