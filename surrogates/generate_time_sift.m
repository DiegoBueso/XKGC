function z = generate_time_sift(x)
    d=length(x);
    T=round(d/3);
    idx=randi([1,d-T]);
    z=x(idx:idx+T);
end