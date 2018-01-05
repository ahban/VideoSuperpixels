% BSD 2-Clause License
% 
% Copyright (c) 2018, Zhihua Ban
% All rights reserved.
% 
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are met:
% 
% * Redistributions of source code must retain the above copyright notice, this
%   list of conditions and the following disclaimer.
% 
% * Redistributions in binary form must reproduce the above copyright notice,
%   this list of conditions and the following disclaimer in the documentation
%   and/or other materials provided with the distribution.
% 
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
% DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
% FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
% DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
% SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
% CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
% OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
% OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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