import os

test_img_path = '../input/test'

filenames = os.listdir(test_img_path)

test_filenames = []

for filename in filenames:
    if not '.jpg' in filename:
        continue
    test_filenames.append(filename)
    
test_txt_file = open('test.txt', 'w')

for test_filename in test_filenames:
    test_txt_file.write('%s\n'%(os.path.abspath(os.path.join(test_img_path,test_filename))))
test_txt_file.close()