import os
paths = ['./dataset/images', './dataset/annotations']

""" def check_make_dir(path):
    if not os.path.exists(path):
        path = path.split('/')[1:]
        for i in range(len(path)):
            tmp_path = path[:i+1]
            if os.path.exists(tmp_path):
                continue
            os.mkdir(tmp_path)
        path = 
         """

path = os.getcwd()
print(path)
path = path.replace('\\', '/')
print(path)