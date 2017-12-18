function [ frame_list, frame_ext, stem_list ] = get_frame_list(seq_root)


frame_ext = get_ext(seq_root);
files = dir(fullfile(seq_root, ['*', frame_ext]));
file_list = cell(numel(files),1);
file_num_parts = cell(numel(files),1);

for i = 1:numel(files)
  file_name = files(i).name;
  file_list{i} = file_name;
  [bg, ed] = regexp(file_name, '[0-9]+');
  numbers = zeros(1, numel(bg));
  for j = 1:numel(bg)
    numbers(j) = str2double( file_name( bg(j):ed(j) ) );
  end
  file_num_parts{i} = numbers;
end

changed_part = (file_num_parts{1}-file_num_parts{end})~=0;

varied_values = zeros(1, numel(file_num_parts));
for i = 1:numel(file_num_parts)
  varied_values(i) = file_num_parts{i}(changed_part);
end

[~, idx] = sort(varied_values, 'ascend');

frame_list = file_list(idx);

stem_list = cell(numel(frame_list),1);
for i = 1:numel(stem_list)
  [~, stem_list{i}, ~] = fileparts(frame_list{i});
end

end


function [ext] = get_ext(seq_root)
  ext_hist = containers.Map;
  files = dir(seq_root);
  for i = 1:numel(files)
    if files(i).isdir
      continue
    end
    [~, ~, ext] = fileparts(files(i).name);
    if ~ext_hist.isKey(ext)
      ext_hist(ext) = 1;
    else
      ext_hist(ext) = ext_hist(ext) + 1;
    end
  end
    
  exts = ext_hist.keys;
  max_count = 0;
  for i = 1:numel(exts)
    if ext_hist(exts{i}) < max_count
      max_count = ext_hist(exts{i});
      ext = exts(i);
    end
  end
end