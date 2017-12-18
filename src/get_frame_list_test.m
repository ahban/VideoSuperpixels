clear
clc

sequence_root = '../sequence/test';

[frame_list, frame_ext, stem_list] = get_frame_list(sequence_root);

for i = 1:numel(frame_list)
  frame_name = frame_list{i};
  frame_stem = stem_list{i};
  frame_full = fullfile(sequence_root, frame_name);
  fprintf('%s\t%s\t%s\n', frame_full, frame_stem, frame_ext);
end