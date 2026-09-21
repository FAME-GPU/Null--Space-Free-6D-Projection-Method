function Xfft = rank1_exact_fft_scatter(x,plan)
%RANK1_EXACT_FFT_SCATTER Scatter retained coefficients and apply 1-D FFT.

    if isvector(x), x=x(:); end
    if size(x,1)~=plan.NF
        error('rank1_exact_fft_scatter:BadSize','x must have plan.NF rows.');
    end

    nrhs=size(x,2);
    xEmbed=complex(zeros(plan.M,nrhs,'like',x));
    xEmbed(plan.inputIdx,:)=x;
    Xfft=fft(xEmbed,[],1);
end
