clear
clc
close all


addpath(genpath('.'))

sequence_root = 'sequence/test';


[frame_list, frame_ext, stem_list] = get_frame_list(sequence_root);

v_s = 20;

figure('units','normalized','outerposition',[0 0 1 1])
for i = 2:numel(frame_list)
  frame_name_1 = fullfile(sequence_root, frame_list{i-1});
  frame_name_2 = fullfile(sequence_root, frame_list{i});
  image_1 = imread(frame_name_1);
  image_2 = imread(frame_name_2);
  labels = mex_vsp(image_1, image_2, v_s);
  labels{1} = labels{1} + 1;
  labels{2} = labels{2} + 1;
  [painted_images, colors] = paint_colors(labels);
    
  subplot(2,2,1); imshow(image_1); title(['Frame ', num2str(i-1)]);
  subplot(2,2,2); imshow(image_2); title(['Frame ', num2str(i)]);
  
  subplot(2,2,3); imshow(painted_images{1});
  subplot(2,2,4); imshow(painted_images{2});
  
  pause(0);
end
