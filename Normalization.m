function m = Normalization(m, type)
    switch type % 不归一化
        case 1
            m=m;
        case 2  %缩放归一化
            Min=min(m,[],2);
            Max=max(m,[],2);
            MM=Max-Min;
            m=(m-Min)./MM;
        case 3  %标准归一化
            m=normalize(m','zscore')';
        case 4  %中心归一化
            m=normalize(m','cente')';
        case 5  %缩放
            Mean=mean(mean(abs(m)));
            m=m/Mean;
        case 6
    end
end