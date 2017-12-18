function colors = get_k_different_colors( K )
% get K unique random colors looks very different



nk = ceil(nthroot(K, 3));

color_elem = ceil(linspace(0, 255, nk+2));
color_elem = color_elem(2:end-1);
colors = zeros(nk*nk*nk,3);
for i = 1:nk
  for j = 1:nk
    for k = 1:nk
      ijk = (i-1)*(nk*nk) + (j-1)*nk + k;
      colors(ijk, :) = [color_elem(i), color_elem(j), color_elem(k)];
    end
  end
end

colors = colors(randperm(size(colors,1)),:);
colors = colors(1:K, :);


end

