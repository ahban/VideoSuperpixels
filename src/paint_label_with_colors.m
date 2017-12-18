function [ color_image ] = paint_label_with_colors( label, colors )

K = max(unique(label));
if K > size(colors, 1)
  disp('less number of colors have been specified');
  return ;
end

[H, W, ~] = size(label);

color_image = zeros(H, W, 3, 'uint8');

for y = 1:H
  for x = 1:W
    lid = label(y, x);
    color_image(y,x,1) = colors(lid, 1);
    color_image(y,x,2) = colors(lid, 2);
    color_image(y,x,3) = colors(lid, 3);    
  end
end


end

