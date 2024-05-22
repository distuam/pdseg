# Copyright (c) OpenMMLab. All rights reserved.
import mmengine.fileio as fileio

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class UQ30T2Dataset(BaseSegDataset): #30个类进行二分类
    classes = (
        "background",
        "plant_diseases")
    METAINFO = dict(
        classes=classes,
        palette=[[i,i,i] for i in range(len(classes))])

    def __init__(self,
                 ann_file='',
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            ann_file=ann_file,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
        assert fileio.exists(self.data_prefix['img_path'], self.backend_args)