"""
Evaluate generated images using Mask2Former (or other object detector model)
"""

import argparse
import json
import os
import re
import sys
import time
import os.path as osp

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import tqdm
import pandas as pd
from PIL import Image, ImageOps
import torch
import mmdet
from mmdet.apis import inference_detector, init_detector

import open_clip
from clip_benchmark.metrics import zeroshot_classification as zsc
zsc.tqdm = lambda it, *args, **kwargs: it


def mmdet_major_version():
    version = getattr(mmdet, "__version__", "0")
    major = version.split(".", 1)[0]
    try:
        return int(major)
    except ValueError:
        return 0

# Get directory path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagedir", type=str)
    parser.add_argument("--outfile", type=str, default="results.jsonl")
    parser.add_argument("--model-config", type=str, default="")
    parser.add_argument("--model-path", type=str, default="")
    # Other arguments
    parser.add_argument("--options", nargs="*", type=str, default=[])
    args = parser.parse_args()
    args.options = dict(opt.split("=", 1) for opt in args.options)
    if args.model_config is None:
        args.model_config = os.path.join(
            os.path.dirname(mmdet.__file__),
            "../configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py"
        )
    return args

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
assert DEVICE == "cuda"

def timed(fn):
    def wrapper(*args, **kwargs):
        startt = time.time()
        result = fn(*args, **kwargs)
        endt = time.time()
        print(f'Function {fn.__name__!r} executed in {endt - startt:.3f}s', file=sys.stderr)
        return result
    return wrapper

# Load models

@timed
def load_models(args):
    CONFIG_PATH = osp.abspath(args.model_config)
    OBJECT_DETECTOR = args.options.get('model', "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco")
    CKPT_PATH = os.path.join(args.model_path, f"{OBJECT_DETECTOR}.pth")
    if mmdet_major_version() >= 3:
        object_detector = init_detector(CONFIG_PATH, None, device=DEVICE)
        load_result = object_detector.load_state_dict(load_compatible_state_dict(CKPT_PATH), strict=False)
        if load_result.missing_keys or load_result.unexpected_keys:
            print(
                "Checkpoint load mismatch after compatibility remap:",
                f"missing={len(load_result.missing_keys)}",
                f"unexpected={len(load_result.unexpected_keys)}",
                file=sys.stderr,
            )
    else:
        object_detector = init_detector(CONFIG_PATH, CKPT_PATH, device=DEVICE)

    clip_arch = args.options.get('clip_model', "ViT-L-14")
    clip_pretrained = os.environ.get("GENEVAL_OPENCLIP_PRETRAINED", "openai")
    clip_model, _, transform = open_clip.create_model_and_transforms(clip_arch, pretrained=clip_pretrained, device=DEVICE)
    tokenizer = open_clip.get_tokenizer(clip_arch)

    with open(os.path.join(os.path.dirname(__file__),'prompts' ,"object_names.txt")) as cls_file:
        classnames = [line.strip() for line in cls_file]

    return object_detector, (clip_model, transform, tokenizer), classnames


def remap_mask2former_checkpoint_key(key):
    if key.startswith("module."):
        key = key[len("module."):]
    key = re.sub(
        r"panoptic_head\.pixel_decoder\.encoder\.layers\.(\d+)\.attentions\.0\.",
        r"panoptic_head.pixel_decoder.encoder.layers.\1.self_attn.",
        key,
    )
    key = re.sub(
        r"panoptic_head\.pixel_decoder\.encoder\.layers\.(\d+)\.ffns\.0\.",
        r"panoptic_head.pixel_decoder.encoder.layers.\1.ffn.",
        key,
    )
    key = re.sub(
        r"panoptic_head\.transformer_decoder\.layers\.(\d+)\.attentions\.0\.",
        r"panoptic_head.transformer_decoder.layers.\1.self_attn.",
        key,
    )
    key = re.sub(
        r"panoptic_head\.transformer_decoder\.layers\.(\d+)\.attentions\.1\.",
        r"panoptic_head.transformer_decoder.layers.\1.cross_attn.",
        key,
    )
    key = re.sub(
        r"panoptic_head\.transformer_decoder\.layers\.(\d+)\.ffns\.0\.",
        r"panoptic_head.transformer_decoder.layers.\1.ffn.",
        key,
    )
    return key


def load_compatible_state_dict(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("state_dict") or checkpoint.get("model") or checkpoint
    else:
        state_dict = checkpoint
    return {
        remap_mask2former_checkpoint_key(key): value
        for key, value in state_dict.items()
    }


def normalize_detector_output(result, num_classes):
    if isinstance(result, tuple):
        bbox = result[0]
        segm = result[1] if len(result) > 1 else None
        return bbox, segm
    if isinstance(result, list):
        return result, None

    if not hasattr(result, "pred_instances"):
        raise TypeError(f"Unsupported detector output type: {type(result)!r}")

    pred_instances = result.pred_instances
    labels = pred_instances.labels.detach().cpu().numpy()
    bboxes = pred_instances.bboxes.detach().cpu().numpy()
    scores = pred_instances.scores.detach().cpu().numpy()
    masks = None
    if hasattr(pred_instances, "masks"):
        masks = pred_instances.masks.detach().cpu().numpy()

    bbox_by_class = []
    segm_by_class = [] if masks is not None else None
    for class_idx in range(num_classes):
        class_mask = labels == class_idx
        if np.any(class_mask):
            class_boxes = np.concatenate(
                [bboxes[class_mask], scores[class_mask, None]],
                axis=1,
            ).astype(np.float32, copy=False)
            bbox_by_class.append(class_boxes)
            if segm_by_class is not None:
                segm_by_class.append(masks[class_mask])
        else:
            bbox_by_class.append(np.zeros((0, 5), dtype=np.float32))
            if segm_by_class is not None:
                segm_by_class.append(np.empty((0,), dtype=bool))
    return bbox_by_class, segm_by_class


COLORS = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white"]
COLOR_CLASSIFIERS = {}

# Evaluation parts

class ImageCrops(torch.utils.data.Dataset):
    def __init__(self, image: Image.Image, objects):
        self._image = image.convert("RGB")
        bgcolor = args.options.get('bgcolor', "#999")
        if bgcolor == "original":
            self._blank = self._image.copy()
        else:
            self._blank = Image.new("RGB", image.size, color=bgcolor)
        self._objects = objects

    def __len__(self):
        return len(self._objects)

    def __getitem__(self, index):
        box, mask = self._objects[index]
        if mask is not None:
            assert tuple(self._image.size[::-1]) == tuple(mask.shape), (index, self._image.size[::-1], mask.shape)
            image = Image.composite(self._image, self._blank, Image.fromarray(mask))
        else:
            image = self._image
        if args.options.get('crop', '1') == '1':
            image = image.crop(box[:4])
        return (transform(image), 0)


def color_classification(image, bboxes, classname):
    if classname not in COLOR_CLASSIFIERS:
        COLOR_CLASSIFIERS[classname] = zsc.zero_shot_classifier(
            clip_model, tokenizer, COLORS,
            [
                f"a photo of a {{c}} {classname}",
                f"a photo of a {{c}}-colored {classname}",
                f"a photo of a {{c}} object"
            ],
            DEVICE
        )
    clf = COLOR_CLASSIFIERS[classname]
    dataloader = torch.utils.data.DataLoader(
        ImageCrops(image, bboxes),
        batch_size=16, num_workers=4
    )
    with torch.no_grad():
        pred, _ = zsc.run_classification(clip_model, clf, dataloader, DEVICE)
        return [COLORS[index.item()] for index in pred.argmax(1)]


def compute_iou(box_a, box_b):
    area_fn = lambda box: max(box[2] - box[0] + 1, 0) * max(box[3] - box[1] + 1, 0)
    i_area = area_fn([
        max(box_a[0], box_b[0]), max(box_a[1], box_b[1]),
        min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    ])
    u_area = area_fn(box_a) + area_fn(box_b) - i_area
    return i_area / u_area if u_area else 0


def relative_position(obj_a, obj_b):
    """Give position of A relative to B, factoring in object dimensions"""
    boxes = np.array([obj_a[0], obj_b[0]])[:, :4].reshape(2, 2, 2)
    center_a, center_b = boxes.mean(axis=-2)
    dim_a, dim_b = np.abs(np.diff(boxes, axis=-2))[..., 0, :]
    offset = center_a - center_b
    revised_offset = np.maximum(np.abs(offset) - POSITION_THRESHOLD * (dim_a + dim_b), 0) * np.sign(offset)
    if np.all(np.abs(revised_offset) < 1e-3):
        return set()
    dx, dy = revised_offset / np.linalg.norm(offset)
    relations = set()
    if dx < -0.5: relations.add("left of")
    if dx > 0.5: relations.add("right of")
    if dy < -0.5: relations.add("above")
    if dy > 0.5: relations.add("below")
    return relations


def evaluate(image, objects, metadata):
    """
    Evaluate given image using detected objects on the global metadata specifications.
    Assumptions:
    * Metadata combines 'include' clauses with AND, and 'exclude' clauses with OR
    * All clauses are independent, i.e., duplicating a clause has no effect on the correctness
    * CHANGED: Color and position will only be evaluated on the most confidently predicted objects;
        therefore, objects are expected to appear in sorted order
    """
    correct = True
    reason = []
    matched_groups = []
    # Check for expected objects
    for req in metadata.get('include', []):
        classname = req['class']
        matched = True
        found_objects = objects.get(classname, [])[:req['count']]
        if len(found_objects) < req['count']:
            correct = matched = False
            reason.append(f"expected {classname}>={req['count']}, found {len(found_objects)}")
        else:
            if 'color' in req:
                # Color check
                colors = color_classification(image, found_objects, classname)
                if colors.count(req['color']) < req['count']:
                    correct = matched = False
                    reason.append(
                        f"expected {req['color']} {classname}>={req['count']}, found " +
                        f"{colors.count(req['color'])} {req['color']}; and " +
                        ", ".join(f"{colors.count(c)} {c}" for c in COLORS if c in colors)
                    )
            if 'position' in req and matched:
                # Relative position check
                expected_rel, target_group = req['position']
                if matched_groups[target_group] is None:
                    correct = matched = False
                    reason.append(f"no target for {classname} to be {expected_rel}")
                else:
                    for obj in found_objects:
                        for target_obj in matched_groups[target_group]:
                            true_rels = relative_position(obj, target_obj)
                            if expected_rel not in true_rels:
                                correct = matched = False
                                reason.append(
                                    f"expected {classname} {expected_rel} target, found " +
                                    f"{' and '.join(true_rels)} target"
                                )
                                break
                        if not matched:
                            break
        if matched:
            matched_groups.append(found_objects)
        else:
            matched_groups.append(None)
    # Check for non-expected objects
    for req in metadata.get('exclude', []):
        classname = req['class']
        if len(objects.get(classname, [])) >= req['count']:
            correct = False
            reason.append(f"expected {classname}<{req['count']}, found {len(objects[classname])}")
    return correct, "\n".join(reason)


def evaluate_image(filepath, metadata):
    result = inference_detector(object_detector, filepath)
    bbox, segm = normalize_detector_output(result, len(classnames))
    image = ImageOps.exif_transpose(Image.open(filepath))
    detected = {}
    # Determine bounding boxes to keep
    confidence_threshold = THRESHOLD if metadata['tag'] != "counting" else COUNTING_THRESHOLD
    for index, classname in enumerate(classnames):
        ordering = np.argsort(bbox[index][:, 4])[::-1]
        ordering = ordering[bbox[index][ordering, 4] > confidence_threshold] # Threshold
        ordering = ordering[:MAX_OBJECTS].tolist() # Limit number of detected objects per class
        detected[classname] = []
        while ordering:
            max_obj = ordering.pop(0)
            detected[classname].append((bbox[index][max_obj], None if segm is None else segm[index][max_obj]))
            ordering = [
                obj for obj in ordering
                if NMS_THRESHOLD == 1 or compute_iou(bbox[index][max_obj], bbox[index][obj]) < NMS_THRESHOLD
            ]
        if not detected[classname]:
            del detected[classname]
    # Evaluate
    is_correct, reason = evaluate(image, detected, metadata)
    return {
        'filename': filepath,
        'tag': metadata['tag'],
        'prompt': metadata['prompt'],
        'correct': is_correct,
        'reason': reason,
        'metadata': json.dumps(metadata),
        'details': json.dumps({
            key: [box.tolist() for box, _ in value]
            for key, value in detected.items()
        })
    }


def main(args):
    full_results = []
    pbar = tqdm.tqdm(total=len(os.listdir(args.imagedir)))
    for subfolder in os.listdir(args.imagedir):
        pbar.update(1)
        folderpath = os.path.join(args.imagedir, subfolder)
        if not os.path.isdir(folderpath) or not subfolder.isdigit():
            print('skip 269')
            continue
        with open(os.path.join(folderpath, "metadata.jsonl")) as fp:
            metadata = json.load(fp)
        # Evaluate each image
        for imagename in os.listdir(os.path.join(folderpath, "samples")):
            imagepath = os.path.join(folderpath, "samples", imagename)
            if not os.path.isfile(imagepath) or not (re.match(r"\d+\.png", imagename) or re.match(r"\d+\.jpg", imagename)):
                print('skip 276')
                continue
            result = evaluate_image(imagepath, metadata)
            full_results.append(result)
    # Save results
    if os.path.dirname(args.outfile):
        os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    with open(args.outfile, "w") as fp:
        pd.DataFrame(full_results).to_json(fp, orient="records", lines=True)


if __name__ == "__main__":
    args = parse_args()
    object_detector, (clip_model, transform, tokenizer), classnames = load_models(args)
    THRESHOLD = float(args.options.get('threshold', 0.3))
    COUNTING_THRESHOLD = float(args.options.get('counting_threshold', 0.9))
    MAX_OBJECTS = int(args.options.get('max_objects', 16))
    NMS_THRESHOLD = float(args.options.get('max_overlap', 1.0))
    POSITION_THRESHOLD = float(args.options.get('position_threshold', 0.1))

    main(args)
