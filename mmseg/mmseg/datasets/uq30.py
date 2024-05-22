# Copyright (c) OpenMMLab. All rights reserved.
import mmengine.fileio as fileio

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class UQ30Dataset(BaseSegDataset):
    classes = (
        "background",
        "maple tar spot",
        "grape downy mildew",
        "plum pocket disease",
        "tomato yellow leaf curl virus",
        "apple black rot",
        "blueberry rust",
        "corn gray leaf spot",
        "cabbage alternaria leaf spot",
        "celery anthracnose",
        "cucumber angular leaf spot",
        "peach leaf curl",
        "citrus canker",
        "rice blast",
        "bean halo blight",
        "tomato late blight",
        "strawberry anthracnose",
        "corn smut",
        "grapevine leafroll disease",
        "garlic leaf blight",
        "tobacco mosaic virus",
        "grape black rot",
        "bell pepper leaf spot",
        "basil downy mildew",
        "ginger leaf spot",
        "coffee leaf rust",
        "cucumber powdery mildew",
        "cauliflower alternaria leaf spot",
        "carrot cavity spot",
        "potato late blight")
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