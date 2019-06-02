import os

py_path = os.path.abspath(__file__)[:-7]

cfg_name = py_path.split('/')[-1]

result_path = os.path.abspath(os.path.join(py_path, '../../result/%s'%(cfg_name)))
darknet_path = os.path.abspath(os.path.join(py_path, '../../darknet/darknet'))
data_path = os.path.abspath(os.path.join(py_path, 'my.data'))
cfg_path = os.path.abspath(os.path.join(py_path, 'my-yolov3-tiny.cfg'))
base_widgets_path = os.path.abspath(os.path.join(py_path, '../../data/darknet53.conv.74'))

if not os.path.exists(result_path):
    os.makedirs(result_path)
    
os.system('%s detector train %s %s %s -dont_show -map'%(
        darknet_path, data_path, cfg_path, base_widgets_path))