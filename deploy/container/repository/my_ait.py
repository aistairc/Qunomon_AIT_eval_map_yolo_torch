#!/usr/bin/env python
# coding: utf-8

# # AIT Development notebook

# ## notebook of structure

# | #  | Name                                               | cells | for_dev | edit               | description                                                                |
# |----|----------------------------------------------------|-------|---------|--------------------|----------------------------------------------------------------------------|
# | 1  | [Environment detection](##1-Environment-detection) | 1     | No      | uneditable         | detect whether the notebook are invoked for packaging or in production     |
# | 2  | [Preparing AIT SDK](##2-Preparing-AIT-SDK)         | 1     | Yes     | uneditable         | download and install AIT SDK                                               |
# | 3  | [Dependency Management](##3-Dependency-Management) | 3     | Yes     | required(cell #2)  | generate requirements.txt for Docker container                             |
# | 4  | [Importing Libraries](##4-Importing-Libraries)     | 2     | Yes     | required(cell #1)  | import required libraries                                                  |
# | 5  | [Manifest Generation](##5-Manifest-Generation)     | 1     | Yes     | required           | generate AIT Manifest                                                      |
# | 6  | [Prepare for the Input](##6-Prepare-for-the-Input) | 1     | Yes     | required           | generate AIT Input JSON (inventory mapper)                                 |
# | 7  | [Initialization](##7-Initialization)               | 1     | No      | uneditable         | initialization for AIT execution                                           |
# | 8  | [Function definitions](##8-Function-definitions)   | N     | No      | required           | define functions invoked from Main area.<br> also define output functions. |
# | 9  | [Main Algorithms](##9-Main-Algorithms)             | 1     | No      | required           | area for main algorithms of an AIT                                         |
# | 10 | [Entry point](##10-Entry-point)                    | 1     | No      | uneditable         | an entry point where Qunomon invoke this AIT from here                     |
# | 11 | [License](##11-License)                            | 1     | Yes     | required           | generate license information                                               |
# | 12 | [Deployment](##12-Deployment)                      | 1     | Yes     | uneditable         | convert this notebook to the python file for packaging purpose             |

# ## notebook template revision history

# 1.0.1 2020/10/21
# 
# * add revision history
# * separate `create requirements and pip install` editable and noeditable
# * separate `import` editable and noeditable
# 
# 1.0.0 2020/10/12
# 
# * new cerarion

# ## body

# ### #1 Environment detection

# [uneditable]

# In[1]:


# Determine whether to start AIT or jupyter by startup argument
import sys
is_ait_launch = (len(sys.argv) == 2)


# ### #2 Preparing AIT SDK

# [uneditable]

# In[2]:


if not is_ait_launch:
    # get ait-sdk file name
    from pathlib import Path
    from glob import glob
    import re
    import os

    current_dir = get_ipython().run_line_magic('pwd', '')

    ait_sdk_path = "./ait_sdk-*-py3-none-any.whl"
    ait_sdk_list = glob(ait_sdk_path)
    ait_sdk_name = os.path.basename(ait_sdk_list[-1])

    # install ait-sdk
    get_ipython().system('pip install -q --upgrade pip')
    get_ipython().system('pip install -q --no-deps --force-reinstall ./$ait_sdk_name')


# ### #3 Dependency Management

# #### #3-1 [uneditable]

# In[3]:


if not is_ait_launch:
    from ait_sdk.common.files.ait_requirements_generator import AITRequirementsGenerator
    requirements_generator = AITRequirementsGenerator()


# #### #3-2 [required]

# In[4]:


if not is_ait_launch:
    requirements_generator.add_package('protobuf','3.19.4')
    requirements_generator.add_package('torch','2.1.1')
    requirements_generator.add_package('yolox','0.3.0')


# #### #3-3 [uneditable]

# In[5]:


if not is_ait_launch:
    requirements_generator.add_package(f'./{ait_sdk_name}')
    requirements_path = requirements_generator.create_requirements(current_dir)

    get_ipython().system('pip install -q -r $requirements_path ')


# ### #4 Importing Libraries

# #### #4-1 [required]

# In[6]:


import numpy as np
import torch
import os
from pathlib import Path

import contextlib
import io
import json
import tempfile
from collections import defaultdict
from tqdm import tqdm

from pycocotools.cocoeval import COCOeval
from yolox.exp import get_exp
from yolox.data import COCODataset, ValTransform
from yolox.utils import (
    postprocess,
    xyxy2xywh
)


# #### #4-2 [uneditable]

# In[7]:


# must use modules
from os import path
import shutil  # do not remove
from ait_sdk.common.files.ait_input import AITInput  # do not remove
from ait_sdk.common.files.ait_output import AITOutput  # do not remove
from ait_sdk.common.files.ait_manifest import AITManifest  # do not remove
from ait_sdk.develop.ait_path_helper import AITPathHelper  # do not remove
from ait_sdk.utils.logging import get_logger, log, get_log_path  # do not remove
from ait_sdk.develop.annotation import measures, resources, downloads, ait_main  # do not remove
# must use modules


# ### #5 Manifest Generation

# [required]

# In[8]:


if not is_ait_launch:
    from ait_sdk.common.files.ait_manifest_generator import AITManifestGenerator
    manifest_generator = AITManifestGenerator(current_dir)
    manifest_generator.set_ait_name('eval_map_yolox_torch')
    manifest_generator.set_ait_description('Evaluate performance of YoloX (Original Pytorch Implementation) model using pycocotools.')
    manifest_generator.set_ait_source_repository('https://github.com/aistairc/eval-map-yolox-torch')
    manifest_generator.set_ait_version('0.1')
    manifest_generator.add_ait_keywords('AIT')
    manifest_generator.add_ait_keywords('Object Detection')
    manifest_generator.add_ait_keywords('YoloX')
    manifest_generator.add_ait_keywords('mAP')
    manifest_generator.set_ait_quality('https://ait-hub.pj.aist.go.jp/ait-hub/api/0.0.1/qualityDimensions/機械学習品質マネジメントガイドライン第三版/C-1機械学習モデルの正確性')
    
    # AIT Inventories
    ## Model Weights
    model_weights_req = manifest_generator.format_ait_inventory_requirement(format_=['pt'])
    manifest_generator.add_ait_inventories(name='torch_weights',
                                             type_='model',
                                             description="Specify state_dict saved through torch.save API. "
                                                         "Assumed architecture of this weights file must match with 'torch_model_module''s one.",
                                             requirement=model_weights_req)
    
    ## Dataset
    ds_req = manifest_generator.format_ait_inventory_requirement(format_=['zip'])
    manifest_generator.add_ait_inventories(name='coco_dataset', 
                                             type_='dataset', 
                                             description='Specify test dataset. They must be conform to the COCO format.',
                                             requirement=ds_req)
    
    # AIT Parameters
    manifest_generator.add_ait_parameters(name='model_name',type_='str',description='Specify model name (Such as yolox-s, yolox-tiny, yolox-nano, ...).', default_val='yolox_s')
    manifest_generator.add_ait_parameters(name='annotation_file_name',type_='str',description='Specify the name of the annotation file in the datasets.zip/annotations directory.', default_val='instaces_val2017.json')
    manifest_generator.add_ait_parameters(name='image_path',type_='str',description='Specify the name of the image directory in the datasets.zip/annotations directory.', default_val='val2017')
    
    manifest_generator.add_ait_parameters(name='image_width',type_='int',description='Image size after preprocess (given to cocoeval)', default_val='640')
    manifest_generator.add_ait_parameters(name='image_height',type_='int',description='Image size after preprocess (given to cocoeval)', default_val='640')
    manifest_generator.add_ait_parameters(name='confthre',type_='float',description='Minimum confidence threshold', default_val='0.01')
    manifest_generator.add_ait_parameters(name='nmsthre',type_='float',description='IoU threshold for NMS process', default_val='0.65')
    manifest_generator.add_ait_parameters(name='num_classes',type_='int',description='Number of classes of detector', default_val='80')
    manifest_generator.add_ait_parameters(name='batch_size',type_='int',description='Specify batch size for the evaluation.', default_val='64')
    
    # AIT Measuers
    manifest_generator.add_ait_measures(name='map',type_='float',description='mAP@[.5 : .05 : 0.95]',structure='single',min='0')
    manifest_generator.add_ait_measures(name='ap_50',type_='float',description='AP@.5',structure='single',min='0')
    manifest_generator.add_ait_measures(name='ap_75',type_='float',description='AP@.75',structure='single',min='0')
    manifest_generator.add_ait_measures(name='map_small',type_='float',description='mAP, with only small (from 0x0 to 32x32) bboxes.',structure='single',min='0')
    manifest_generator.add_ait_measures(name='map_medium',type_='float',description='mAP, with only medium (from 32x32 to 96x96) bboxes.',structure='single',min='0')
    manifest_generator.add_ait_measures(name='map_large',type_='float',description='mAP, with only large (from 96x96 to 10000x10000) bboxes.',structure='single',min='0')
    manifest_generator.add_ait_measures(name='mrec_maxdet1',type_='float',description='mRecall[0 : 0.01 : 1] at max_det==1',structure='single',min='0')
    manifest_generator.add_ait_measures(name='mrec_maxdet10',type_='float',description='mRecall[0 : 0.01 : 1] at max_det==10',structure='single',min='0')
    manifest_generator.add_ait_measures(name='mrec_maxdet100',type_='float',description='mRecall[0 : 0.01 : 1] at max_det==100',structure='single',min='0')
    manifest_generator.add_ait_measures(name='mrec_small',type_='float',description='mRecall, with only small (from 0x0 to 32x32) bboxes.',structure='single',min='0')
    manifest_generator.add_ait_measures(name='mrec_medium',type_='float',description='mRecall, with only medium (from 32x32 to 96x96) bboxes.',structure='single',min='0')
    manifest_generator.add_ait_measures(name='mrec_large',type_='float',description='mRecall, with only large (from 96x96 to 10000x10000) bboxes.',structure='single',min='0')

    manifest_generator.add_ait_resources(name='summarized_text',type_='text',description='summarized results (cocoeval output)')

    manifest_generator.add_ait_downloads(name='Log',description='AIT実行ログ')

    manifest_path = manifest_generator.write()


# ### #6 Prepare for the Input

# [required]

# In[9]:


if not is_ait_launch:
    from ait_sdk.common.files.ait_input_generator import AITInputGenerator
    input_generator = AITInputGenerator(manifest_path)
    input_generator.add_ait_inventories(name='coco_dataset',
                                        value='coco_dataset/dataset.zip')
    input_generator.add_ait_inventories(name='torch_weights',
                                        value='torch_weights/yolox_s.pth')
    input_generator.set_ait_params(name='model_name',value='yolox_s')
    input_generator.set_ait_params(name='annotation_file_name',value='instances_valreduced.json')
    input_generator.set_ait_params(name='image_path',value='valreduced')
    input_generator.set_ait_params(name='image_width',value='640')
    input_generator.set_ait_params(name='image_height',value='640')
    input_generator.set_ait_params(name='confthre',value='0.01')
    input_generator.set_ait_params(name='nmsthre',value='0.65')
    input_generator.set_ait_params(name='num_classes',value='80')
    input_generator.set_ait_params(name='batch_size',value='16')
    input_generator.write()


# ### #7 Initialization

# [uneditable]

# In[10]:


logger = get_logger()

ait_manifest = AITManifest()
ait_input = AITInput(ait_manifest)
ait_output = AITOutput(ait_manifest)

if is_ait_launch:
    # launch from AIT
    current_dir = path.dirname(path.abspath(__file__))
    path_helper = AITPathHelper(argv=sys.argv, ait_input=ait_input, ait_manifest=ait_manifest, entry_point_dir=current_dir)
else:
    # launch from jupyter notebook
    # ait.input.json make in input_dir
    input_dir = '/usr/local/qai/mnt/ip/job_args/1/1'
    current_dir = get_ipython().run_line_magic('pwd', '')
    path_helper = AITPathHelper(argv=['', input_dir], ait_input=ait_input, ait_manifest=ait_manifest, entry_point_dir=current_dir)

ait_input.read_json(path_helper.get_input_file_path())
ait_manifest.read_json(path_helper.get_manifest_file_path())

### do not edit cell


# ### #8 Function definitions

# [required]

# In[11]:


# This cell is a modified version of the YoloX Code.
# Original Copyright Notice is shown below.
# Detailed license information is placed at repository root and will be included in the AIT package.

# Copyright (c) Megvii, Inc. and its affiliates.

class CPUCOCOEvaluator:
    def __init__(
        self,
        dataloader,
        img_size: int,
        confthre: float,
        nmsthre: float,
        num_classes: int
    ):
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes

    def evaluate(
        self, model
    ):
        model = model.eval()
        ids = []
        data_list = []
        output_data = defaultdict()
        progress_bar = tqdm

        for imgs, _, info_imgs, ids in progress_bar(self.dataloader):
            with torch.no_grad():
                outputs = model(imgs)

                outputs = postprocess(
                    outputs, self.num_classes, self.confthre, self.nmsthre
                )

            data_list_elem, image_wise_data = self.convert_to_coco_format(
                outputs, info_imgs, ids, return_outputs=True)
            data_list.extend(data_list_elem)
            output_data.update(image_wise_data)

        eval_results = self.evaluate_prediction(data_list)

        return eval_results

    def convert_to_coco_format(self, outputs, info_imgs, ids, return_outputs=False):
        data_list = []
        image_wise_data = defaultdict(dict)
        for (output, img_h, img_w, img_id) in zip(
            outputs, info_imgs[0], info_imgs[1], ids
        ):
            if output is None:
                continue
            output = output.cpu()

            bboxes = output[:, 0:4]

            # preprocessing: resize
            scale = min(
                self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
            )
            bboxes /= scale
            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]

            image_wise_data.update({
                int(img_id): {
                    "bboxes": [box.numpy().tolist() for box in bboxes],
                    "scores": [score.numpy().item() for score in scores],
                    "categories": [
                        self.dataloader.dataset.class_ids[int(cls[ind])]
                        for ind in range(bboxes.shape[0])
                    ],
                }
            })

            bboxes = xyxy2xywh(bboxes)

            for ind in range(bboxes.shape[0]):
                label = self.dataloader.dataset.class_ids[int(cls[ind])]
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)

        if return_outputs:
            return data_list, image_wise_data
        return data_list

    def evaluate_prediction(self, data_dict):
        info = ""
        
        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            cocoGt = self.dataloader.dataset.coco
            
            _, tmp = tempfile.mkstemp()
            json.dump(data_dict, open(tmp, "w"))
            cocoDt = cocoGt.loadRes(tmp)

            cocoEval = COCOeval(cocoGt, cocoDt, "bbox")            
            cocoEval.evaluate()
            cocoEval.accumulate()
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()
            info += redirect_string.getvalue()
            
            return cocoEval, info
        else:
            return None, info


# In[12]:


def gather_metric(ceval, p, metric_type, iou=None, area_range="all", max_det=100):
    evalobj = ceval[metric_type]
    a_indices = [i for i, arl in enumerate(p.areaRngLbl) if area_range == arl]
    d_indices = [i for i, mdets in enumerate(p.maxDets) if max_det == mdets]

    if metric_type == "precision":
        if iou is None:
            resobj = evalobj[:,:,:,a_indices, d_indices]
            result = np.mean(resobj[resobj> -1])
            return float(result)
        else:
            mask = np.where(iou == p.iouThrs)[0]
            resobj = evalobj[mask][:,:,:,a_indices, d_indices]
            result = np.mean(resobj[resobj> -1])
            return float(result)
    
    elif metric_type == "recall":
        if iou is None:
            resobj = evalobj[:,:,a_indices, d_indices]
            result = np.mean(resobj[resobj> -1])
            return float(result)
        else:
            mask = np.where(iou == p.iouThrs)[0]
            resobj = evalobj[mask][:,:,a_indices, d_indices]
            result = np.mean(resobj[resobj> -1])
            return float(result)
        
    else:
        raise NotImplementedError()
    


# In[13]:


@log(logger)
@measures(ait_output, 'map')
def calc_map(cocoeval, p):
    return gather_metric(cocoeval, p, "precision")


# In[14]:


@log(logger)
@measures(ait_output, 'ap_50')
def calc_ap_50(cocoeval, p):
    return gather_metric(cocoeval, p, "precision", iou=.50)


# In[15]:


@log(logger)
@measures(ait_output, 'ap_75')
def calc_ap_75(cocoeval, p):
    return gather_metric(cocoeval, p, "recall", iou=.75)


# In[16]:


@log(logger)
@measures(ait_output, 'map_small')
def calc_map_small(cocoeval, p):
    return gather_metric(cocoeval, p, "recall", area_range="small")


# In[17]:


@log(logger)
@measures(ait_output, 'map_medium')
def calc_map_medium(cocoeval, p):
    return gather_metric(cocoeval, p, "recall", area_range="medium")


# In[18]:


@log(logger)
@measures(ait_output, 'map_large')
def calc_map_large(cocoeval, p):
    return gather_metric(cocoeval, p, "recall", area_range="large")


# In[19]:


@log(logger)
@measures(ait_output, 'mrec_maxdet1')
def calc_mrec_maxdet1(cocoeval, p):
    return gather_metric(cocoeval, p, "recall", max_det=1)


# In[20]:


@log(logger)
@measures(ait_output, 'mrec_maxdet10')
def calc_mrec_maxdet10(cocoeval, p):
    return gather_metric(cocoeval, p, "recall", max_det=10)


# In[21]:


@log(logger)
@measures(ait_output, 'mrec_maxdet100')
def calc_mrec_maxdet100(cocoeval, p):
    return gather_metric(cocoeval, p, "recall", max_det=100)


# In[22]:


@log(logger)
@measures(ait_output, 'mrec_small')
def calc_mrec_small(cocoeval, p):
    return gather_metric(cocoeval, p, "recall", area_range="small")


# In[23]:


@log(logger)
@measures(ait_output, 'mrec_medium')
def calc_mrec_medium(cocoeval, p):
    return gather_metric(cocoeval, p, "recall", area_range="medium")


# In[24]:


@log(logger)
@measures(ait_output, 'mrec_large')
def calc_mrec_large(cocoeval, p):
    return gather_metric(cocoeval, p, "recall", area_range="large")


# In[25]:


@log(logger)
@resources(ait_output, path_helper, 'summarized_text', "summary.txt")
def save_summarized_text(summary, file_path: str=None): 
    os.makedirs(str(Path(file_path).parent), exist_ok=True)
    with open(file_path, "w") as f:
        f.write(summary)


# In[26]:


@log(logger)
@downloads(ait_output, path_helper, 'Log', 'ait.log')
def move_log(file_path: str=None):
    shutil.move(get_log_path(), file_path)


# ### #9 Main Algorithms

# [required]

# In[27]:


@log(logger)
@ait_main(ait_output, path_helper)
def main() -> None:    
    # Prepare fixed parameters
    # TODO: Parameterize these values
    img_w = ait_input.get_method_param_value('image_width')
    img_h = ait_input.get_method_param_value('image_height')
    img_size = (img_w, img_h)
    
    conf_thre = ait_input.get_method_param_value('confthre')
    nms_thre = ait_input.get_method_param_value('nmsthre')
    num_classes = ait_input.get_method_param_value('num_classes')
    
    # Prepare for the model
    model_name = ait_input.get_method_param_value('model_name')
    exp = get_exp(None, model_name)
    model = exp.get_model()
    
    model_weights_path = ait_input.get_inventory_path('torch_weights')
    model_weights = torch.load(model_weights_path, map_location="cpu")
    model.load_state_dict(model_weights["model"])
    
    # load dataset
    ds_path = ait_input.get_inventory_path('coco_dataset')
    annotation_file_name = ait_input.get_method_param_value('annotation_file_name')
    image_path = ait_input.get_method_param_value('image_path')
    batch_size = ait_input.get_method_param_value('batch_size')
    
    if os.path.exists("./temp"):
        shutil.rmtree("./temp")
    os.makedirs("./temp", exist_ok=True)
    pre_dirs = set(os.listdir("./temp"))
    shutil.unpack_archive(ds_path, extract_dir="./temp")
    post_dirs = set(os.listdir("./temp"))
    extracted = list(pre_dirs ^ post_dirs)[0]
    
    unpacked_dir_name = f"./temp/{extracted}"
    
    ds = COCODataset(
        data_dir=f"{unpacked_dir_name}",
        json_file=annotation_file_name,
        name=image_path,
        img_size=img_size,
        preproc=ValTransform()
    )
 
    ds2 = ds
    
#   code for testing with partial dataset.
    ds2 = torch.utils.data.Subset(ds, range(32))
    
    sampler = torch.utils.data.SequentialSampler(ds2)
    loader_kwargs = {
        "num_workers": 0,
        "pin_memory": True,
        "sampler": sampler,
        "batch_size": batch_size
    }
    ds_loader = torch.utils.data.DataLoader(ds, **loader_kwargs)

    evaluator = CPUCOCOEvaluator(
        dataloader=ds_loader,
        img_size=img_size,
        confthre=conf_thre,
        nmsthre=nms_thre,
        num_classes=num_classes)
    
    ceval, summarized_text = evaluator.evaluate(model)
    
    save_summarized_text(summarized_text)
    
    calc_map(ceval.eval, ceval.params)
    calc_ap_50(ceval.eval, ceval.params)
    calc_ap_75(ceval.eval, ceval.params)
    calc_map_small(ceval.eval, ceval.params)
    calc_map_medium(ceval.eval, ceval.params)
    calc_map_large(ceval.eval, ceval.params)
    calc_mrec_maxdet1(ceval.eval, ceval.params)
    calc_mrec_maxdet10(ceval.eval, ceval.params)
    calc_mrec_maxdet100(ceval.eval, ceval.params)
    calc_mrec_small(ceval.eval, ceval.params)
    calc_mrec_medium(ceval.eval, ceval.params)
    calc_mrec_large(ceval.eval, ceval.params)
    
    move_log()


# ### #10 Entry point

# [uneditable]

# In[ ]:


if __name__ == '__main__':
    main()


# ### #11 License

# [required]

# In[ ]:


## sample ##
ait_owner='AIST'
ait_creation_year='2023'


# ### #12 Deployment

# [uneditable] 

# In[ ]:


if not is_ait_launch:
    from ait_sdk.deploy import prepare_deploy
    from ait_sdk.license.license_generator import LicenseGenerator
    
    current_dir = get_ipython().run_line_magic('pwd', '')
    prepare_deploy(ait_sdk_name, current_dir, requirements_path)
    
    # output License.txt
    license_generator = LicenseGenerator()
    license_generator.write('../top_dir/LICENSE.txt', ait_creation_year, ait_owner)


# In[ ]:




