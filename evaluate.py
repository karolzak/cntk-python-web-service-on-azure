
from __future__ import print_function
import numpy as np
import os, sys
import cv2
from cntk import load_model, Axis, input_variable
from cntk.core import Value
from cntk.io import MinibatchData
from cntk.layers import Constant

from utils.annotations.annotations_helper import parse_class_map_file
from config import cfg
from plot_helpers import visualizeResultsFaster, imsave, apply_nms_to_single_image_results
from cntk_helpers import regress_rois

###############################################################
# Variables
###############################################################

image_width = cfg["CNTK"].IMAGE_WIDTH
image_height = cfg["CNTK"].IMAGE_HEIGHT
num_channels = cfg["CNTK"].NUM_CHANNELS

# dims_input -- (pad_width, pad_height, scaled_image_width, scaled_image_height, orig_img_width, orig_img_height)
dims_input_const = MinibatchData(Value(batch=np.asarray(
    [image_width, image_height, image_width, image_height, image_width, image_height], dtype=np.float32)), 1, 1, False)

# Color used for padding and normalization (Caffe model uses [102.98010, 115.94650, 122.77170])
img_pad_value = [103, 116, 123] if cfg["CNTK"].BASE_MODEL == "VGG16" else [114, 114, 114]
normalization_const = Constant([[[103]], [[116]], [[123]]]) if cfg["CNTK"].BASE_MODEL == "VGG16" else Constant([[[114]], [[114]], [[114]]])


globalvars = {}

map_file_path = cfg["CNTK"].MODEL_DIRECTORY
globalvars['class_map_file'] = os.path.join(map_file_path, cfg["CNTK"].CLASS_MAP_FILE)
globalvars['classes'] = parse_class_map_file(globalvars['class_map_file'])
globalvars['num_classes'] = len(globalvars['classes'])
globalvars['temppath'] = cfg["CNTK"].TEMP_PATH
feature_node_name = cfg["CNTK"].FEATURE_NODE_NAME
model_path = os.path.join(cfg["CNTK"].MODEL_DIRECTORY, cfg["CNTK"].MODEL_NAME)

# helper function
def load_resize_and_pad(image_path, width, height, pad_value=114):
    if "@" in image_path:
        print("WARNING: zipped image archives are not supported for visualizing results.")
        exit(0)

    img = cv2.imread(image_path)
    img_width = len(img[0])
    img_height = len(img)
    scale_w = img_width > img_height
    target_w = width
    target_h = height

    if scale_w:
        target_h = int(np.round(img_height * float(width) / float(img_width)))
    else:
        target_w = int(np.round(img_width * float(height) / float(img_height)))

    resized = cv2.resize(img, (target_w, target_h), 0, 0, interpolation=cv2.INTER_NEAREST)

    top = int(max(0, np.round((height - target_h) / 2)))
    left = int(max(0, np.round((width - target_w) / 2)))
    bottom = height - top - target_h
    right = width - left - target_w
    resized_with_pad = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                          cv2.BORDER_CONSTANT, value=[pad_value, pad_value, pad_value])

    # transpose(2,0,1) converts the image to the HWC format which CNTK accepts
    model_arg_rep = np.ascontiguousarray(np.array(resized_with_pad, dtype=np.float32).transpose(2, 0, 1))

    dims = (width, height, target_w, target_h, img_width, img_height)
    return resized_with_pad, model_arg_rep, dims


# mode="returnimage" or "returntags"
def eval_faster_rcnn(eval_model, imgPath, img_shape,
                              results_base_path, feature_node_name, classes, mode,
                              drawUnregressedRois=False, drawNegativeRois=False,
                              nmsThreshold=0.5, nmsConfThreshold=0.0, bgrPlotThreshold = 0.8):

    # prepare model
    image_input = input_variable(img_shape, dynamic_axes=[Axis.default_batch_axis()], name=feature_node_name)
    dims_input = input_variable((1,6), dynamic_axes=[Axis.default_batch_axis()], name='dims_input')
    frcn_eval = eval_model(image_input, dims_input)

    #dims_input_const = cntk.constant([image_width, image_height, image_width, image_height, image_width, image_height], (1, 6))
    print("Plotting results from Faster R-CNN model for image.")
    # evaluate single image

    _, cntk_img_input, dims = load_resize_and_pad(imgPath, img_shape[2], img_shape[1])

    dims_input = np.array(dims, dtype=np.float32)
    dims_input.shape = (1,) + dims_input.shape
    output = frcn_eval.eval({frcn_eval.arguments[0]: [cntk_img_input], frcn_eval.arguments[1]: dims_input})

    out_dict = dict([(k.name, k) for k in output])
    out_cls_pred = output[out_dict['cls_pred']][0]
    out_rpn_rois = output[out_dict['rpn_rois']][0]
    out_bbox_regr = output[out_dict['bbox_regr']][0]

    labels = out_cls_pred.argmax(axis=1)
    scores = out_cls_pred.max(axis=1).tolist()

    if mode=="returntags":
        class Tag(object):
            def __init__(self, label, score, bbox):
                self.label = label
                self.score = score
                self.bbox = bbox

            def serialize(self):
                return {
                    'label': self.label,
                    'score': self.score,
                    'bbox': self.bbox,
                }

        results = []
        for i in range(len(out_rpn_rois)):
            if labels[i] != 0:
                x = Tag(str(classes[labels[i]]), str(scores[i]), str(out_rpn_rois[i]))
                results.append(x)

        return results


    elif mode=="returnimage":
        evaluated_image_path = "{}/{}".format(results_base_path, 'evaluated_' + os.path.basename(imgPath))
        if drawUnregressedRois:
            # plot results without final regression
            imgDebug = visualizeResultsFaster(imgPath, labels, scores, out_rpn_rois, img_shape[2], img_shape[1],
                                              classes, nmsKeepIndices=None, boDrawNegativeRois=drawNegativeRois,
                                              decisionThreshold=bgrPlotThreshold)
            imsave(evaluated_image_path, imgDebug)
        else:
            # apply regression and nms to bbox coordinates
            regressed_rois = regress_rois(out_rpn_rois, out_bbox_regr, labels, dims)

            nmsKeepIndices = apply_nms_to_single_image_results(regressed_rois, labels, scores,
                                                               nms_threshold=nmsThreshold,
                                                               conf_threshold=nmsConfThreshold)

            img = visualizeResultsFaster(imgPath, labels, scores, regressed_rois, img_shape[2], img_shape[1],
                                         classes, nmsKeepIndices=nmsKeepIndices,
                                         boDrawNegativeRois=drawNegativeRois,
                                         decisionThreshold=bgrPlotThreshold)
            imsave(evaluated_image_path, img)

        return evaluated_image_path
    else:
        raise ValueError("Unsupported value found in 'mode' parameter")





# mode="returnimage" or "returntags"
def evaluateimage(file_path, mode, eval_model=None):

    #from plot_helpers import eval_and_plot_faster_rcnn
    if eval_model==None:
        print("Loading existing model from %s" % model_path)
        eval_model = load_model(model_path)
    img_shape = (num_channels, image_height, image_width)
    results_folder = globalvars['temppath']
    results=eval_faster_rcnn(eval_model, file_path, img_shape,
                              results_folder, feature_node_name, globalvars['classes'], mode,
                              drawUnregressedRois=cfg["CNTK"].DRAW_UNREGRESSED_ROIS,
                              drawNegativeRois=cfg["CNTK"].DRAW_NEGATIVE_ROIS,
                              nmsThreshold=cfg["CNTK"].RESULTS_NMS_THRESHOLD,
                              nmsConfThreshold=cfg["CNTK"].RESULTS_NMS_CONF_THRESHOLD,
                              bgrPlotThreshold=cfg["CNTK"].RESULTS_BGR_PLOT_THRESHOLD)
    return results

