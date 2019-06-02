import os
import sys

cfg_path = os.path.abspath(sys.argv[1])
cfg_name = cfg_path.split('/')[-1]
test_txt_path = os.path.abspath(os.path.join(os.path.dirname(__file__),  'test.txt'))


darknet_path = os.path.abspath(os.path.join(cfg_path, '../../darknet/darknet'))
data_path = os.path.abspath(os.path.join(cfg_path, 'my.data'))
best_widgets_path = os.path.abspath(os.path.join(cfg_path, 'my-yolov3-tiny_best.weights'))
result_txt_path = os.path.abspath(os.path.join(cfg_path, 'results.txt'))
for filename in os.listdir(cfg_path):
    if '.cfg' in filename:
        cfg_path = os.path.abspath(os.path.join(cfg_path, filename))

os.system('%s detector test %s %s %s -dont_show -ext_output < %s > %s'%(
        darknet_path, data_path, cfg_path, best_widgets_path, test_txt_path, result_txt_path))