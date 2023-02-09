#tvpr2 的color部分检测
import glob
import re
import os
import os.path as osp

from .bases import BaseImageDataset


class TVPR2(BaseImageDataset):
    def __init__(self,root=r'./', verbose=True, **kwargs):
        super(TVPR2, self).__init__()
        # data_dir = '/data3/QK/REID/small1000'
        data_dir = '/data3/QK/REID/pair10001'
        train_color_list = []###train的彩色图片的路径集合
        train_depth_list = []

        gallery_color_list = []
        gallery_depth_list = []
        query_color_list = []
        query_depth_list = []

        dirs = []
        for dir in os.listdir(data_dir):
            dirs.append(dir)
        dirs.sort(key=lambda x: int(x))
        #读取训练集
        for i in range(0, len(dirs) // 2 + 1):
            path0 = os.path.join(data_dir, dirs[i])  ####pair1000/12
            for file in os.listdir(path0):
                if file.endswith("color.png"):
                    path = os.path.join(path0, file)
                    train_color_list.append(path)
                elif file.endswith("depth.png"):
                    path = os.path.join(path0, file)
                    train_depth_list.append(path)
        #读取gallery和query
        for i in range(len(dirs) // 2 + 1, len(dirs)):
            # print(i)
            # print("dirs[i]")
            # print(dirs[i])
            path0 = os.path.join(data_dir, dirs[i])  ####pair1000/12
            files = []
            for file in os.listdir(path0):
                files.append(file)
            files.sort(key=lambda x: int(x.split('_')[0]))
            for j in range(0, (len(files) // 4) * 2):
                if files[j].endswith("color.png"):
                    path = os.path.join(path0, files[j])
                    gallery_color_list.append(path)
                elif files[j].endswith("depth.png"):
                    path = os.path.join(path0, files[j])
                    gallery_depth_list.append(path)
            for j in range((len(files) // 4) * 2, len(files)):
                if files[j].endswith("color.png"):
                    path = os.path.join(path0, files[j])
                    query_color_list.append(path)
                elif files[j].endswith("depth.png"):
                    path = os.path.join(path0, files[j])
                    query_depth_list.append(path)
        self.train=self._process_dir(train_color_list)
        self.query=self._process_dir(query_color_list)
        self.gallery=self._process_dir(gallery_color_list,is_gallery=True)

        if verbose:
            print("=> TVPR2 loaded")
            self.print_dataset_statistics(self.train, self.query, self.gallery)

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)
    #暂时不考虑映射
    def _process_dir(self, data_file_list,is_gallery=False):
        dataset = []
        for data_path in data_file_list:
            file_label=int(data_path.split('/')[-2])
            # if file_label > 491:  # 把测试集的序号从改成从0开始
            if file_label > 493:
                # file_label = file_label - 492
                file_label = file_label - 494
            if is_gallery:#因为只有一个相机，为了后续的eval_reid 不报错，直接手动修改cam_id
                dataset.append((data_path,file_label,1))
            else:
                dataset.append((data_path, file_label, 0))

        return dataset
