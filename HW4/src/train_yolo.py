import os
import sys

cfg_path = os.path.abspath(sys.argv[1])
cfg_name = cfg_path.split('/')[-1]

darknet_path = os.path.abspath(os.path.join(cfg_path, '../../darknet/darknet'))
data_path = os.path.abspath(os.path.join(cfg_path, 'my.data'))
base_widgets_path = os.path.abspath(os.path.join(cfg_path, '../../data/darknet53.conv.74'))
for filename in os.listdir(cfg_path):
    if '.cfg' in filename:
        cfg_path = os.path.abspath(os.path.join(cfg_path, filename))

os.system('%s detector train %s %s %s -dont_show -map'%(
        darknet_path, data_path, cfg_path, base_widgets_path))