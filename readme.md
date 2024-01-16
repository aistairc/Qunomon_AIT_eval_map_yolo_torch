# eval_map_yolox_torch_0.1
## description
- This AIT evaluates YOLOX models performance with COCO-formtted test data using pycocotools.
- Input
  - PyTorch weights trained for YOLOX model.
  - COCO-formatted datasets.
- Output
  - 12 evaluation metrics (mAP, mRecall) originally provided by the pycocotools (cocoeval).
  - Textual output originally provided by the pycocotools (cocoeval).

## License Notification
- This project is licensed under Apache License Version 2.0. See the `./LICENSE.txt` for more details.
- This project includes the work from YOLOX project, which is originally licensed under the Apache License Version 2.0.
  - The code is modified by the author of this project, and will be explicitly notified at the location of diversion.
  - Original copyright notice and the license text is reproduced at `./ThirdPartyNotices.txt`.