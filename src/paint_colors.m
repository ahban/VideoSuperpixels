
function [painted_images, colors ] = paint_colors(labels)
% randome k different colors

if ~iscell(labels)
  disp('Only support cell labels');
  return ;
end

N = numel(labels);  
Ks = zeros(N,1);  
[H, W, Z] = size(labels{1});

Ku = cell(N,1);
for i = 1:N
  Ku{i} = unique(labels{i});
end
Ka = [];
for i = 1:N
  Ka = [Ka; Ku{i}];
end

K = double(max(Ka));

colors = uint8(get_k_different_colors(K));

% painting
painted_images = cell(N, 1);
for i = 1:N
  painted_images{i} = zeros(H, W, 3, 'uint8');
end

for i = 1:N
  for r = 1:H
    for c = 1:W
      labid = labels{i}(r,c);
      painted_images{i}(r,c,1) = colors(labid, 1);
      painted_images{i}(r,c,2) = colors(labid, 2);
      painted_images{i}(r,c,3) = colors(labid, 3);
    end
  end
end

end