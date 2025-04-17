import torch


def iou(boxes1, boxes2):
    '''
    Compute intersection over union between two sets of bounding boxes in [xmin, ymin, xmax, ymax] form.

    Input:
    boxes1: (Tensor) M x 4
    boxes2: (Tensor) N x 4

    Output:
    jaccard: M x N tensor of IoU values corresponding to each possible pair between the two input sets.
    '''
    inter = intersection(boxes1, boxes2) # M x N
    uni = union(boxes1, boxes2, inter) # M x N
    return inter/uni

def intersection(boxes1, boxes2):
    '''
    Compute intersection between two sets of bounding boxes in [xmin, ymin, xmax, ymax] form.

    Input:
    boxes1: (Tensor) M x 4
    boxes2: (Tensor) N x 4

    Output:
    inter: (Tensor) M x N tensor of intersection values corresponding to each pair between input sets.
    '''
    M = boxes1.size(0)
    N = boxes2.size(0)
    # expand for tensor math
    expanded1_mins = boxes1[:, :2].unsqueeze(1).expand(M, N, 2) # Mx2 -> Mx1x2 -> MxNx2 (xmin and ymin)
    expanded2_mins = boxes2[:, :2].unsqueeze(0).expand(M, N, 2) # Nx2 -> 1xNx2 -> MxNx2
    expanded1_maxes = boxes1[:, 2:].unsqueeze(1).expand(M, N, 2) # xmax and ymax
    expanded2_maxes = boxes2[:, 2:].unsqueeze(0).expand(M, N, 2)

    xmin = torch.max(expanded1_mins[:, :, 0], expanded2_mins[:, :, 0])
    ymin = torch.max(expanded1_mins[:, :, 1], expanded2_mins[:, :, 1])
    xmax = torch.min(expanded1_maxes[:, :, 0], expanded2_maxes[:, :, 0])
    ymax = torch.min(expanded1_maxes[:, :, 1], expanded2_maxes[:, :, 1])

    inter = torch.clamp(xmax-xmin, min=0)*torch.clamp(ymax-ymin, min=0)
    return inter

def union(boxes1, boxes2, inter):
    '''
    Compute union between two sets of bounding boxes in [xmin, ymin, xmax, ymax] form.

    Input:
    boxes1: (Tensor) M x 4
    boxes2: (Tensor) N x 4
    inter: (Tensor) M x N tensor of intersection values corresponding to each pair between input sets.

    Output:
    uni: (Tensor) M x N tensor of union values corresponding to each pair between input sets.
    '''
    M, N = boxes1.size(0), boxes2.size(0)
    areas1 = ((boxes1[:, 2]-boxes1[:, 0])*(boxes1[:, 3]-boxes1[:, 1])).unsqueeze(1).expand(M, N)
    areas2 = ((boxes2[:, 2]-boxes2[:, 0])*(boxes2[:, 3]-boxes2[:, 1])).unsqueeze(0).expand(M, N)
    uni = areas1 + areas2 - inter
    return uni

def precision(y_hat, y):
    '''
    Computes precision between binary prediction mask y_hat and ground truth binary mask y.

    Input:
    y_hat: (Tensor) binary mask of predictions
    y: (Tensor) binary mask of ground-truths

    Output:
    precision: (float) precision score between 0 and 1
    '''
    true_positives = torch.sum(torch.logical_and(y_hat, y)).item()
    all_positives = torch.sum(y_hat).item() # all positives from prediction masks
    return true_positives/all_positives

def recall(y_hat, y):
    '''
    Computes recall between binary prediction mask y_hat and ground truth binary mask y.

    Input:
    y_hat: (Tensor) binary mask of predictions
    y: (Tensor) binary mask of ground-truths

    Output:
    recall: (float) recall score between 0 and 1
    '''
    true_positives = torch.sum(torch.logical_and(y_hat, y)).item()
    all_positives = torch.sum(y).item() # all positives from ground truth masks
    return true_positives/all_positives

def f_measure(y_hat, y):
    '''
    Computes F-measure between binary predictions mask y_hat and ground truth binary mask y.

    Input:
    y_hat: (Tensor) binary mask of predictions
    y: (Tensor) binary mask of ground-truths

    Output:
    f_measure: (float) F-measure score between 0 and 1
    '''
    pre = precision(y_hat, y)
    rec = recall(y_hat, y)
    return 2*pre*rec/(pre+rec)

def calculate_AP(det_boxes, det_labels, det_scores, true_boxes, true_labels, n_classes, T=0.5):
    '''
    Taken from https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/utils.py.

    Calculate the Mean Average Precision (mAP) of detected objects.

    See https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173 for an explanation

    Input:
    det_boxes: (list of tensors) one tensor for each image containing detected objects' bounding boxes
    det_labels: (list of tensors) one tensor for each image containing detected objects' labels
    det_scores: (list of tensors) one tensor for each image containing detected objects' labels' scores
    true_boxes: (list of tensors) one tensor for each image containing actual objects' bounding boxes
    true_labels: (list of tensors) one tensor for each image containing actual objects' labels
    n_classes: (int) total number of classes
    T: (float) IoU threshold for matching predicted boxes to ground-truth boxes, default=0.5

    Output:
    average_precisions: list of average precisions for all classes
    mean_average_precision: (float) average precision (AP) at IoU threshold of T across all classes
    '''
    # these are all lists of tensors of the same length, i.e. number of images
    assert len(det_boxes) == len(det_labels) == len(det_scores) == len(true_boxes) == len(true_labels)

    device = 'cpu'
    # Store all (true) objects in a single continuous tensor while keeping track of the image it is from
    true_images = list()
    for i in range(len(true_labels)):
        true_images.extend([i] * true_labels[i].size(0))
    true_images = torch.LongTensor(true_images).to(device)  # (n_objects), n_objects is the total no. of objects across all images
    true_boxes = torch.cat(true_boxes, dim=0)  # (n_objects, 4)
    true_labels = torch.cat(true_labels, dim=0)  # (n_objects)
    true_difficulties = torch.zeros_like(true_labels)  # (n_objects)
    assert true_images.size(0) == true_boxes.size(0) == true_labels.size(0)

    # Store all detections in a single continuous tensor while keeping track of the image it is from
    det_images = list()
    for i in range(len(det_labels)):
        det_images.extend([i] * det_labels[i].size(0))
    det_images = torch.LongTensor(det_images).to(device)  # (n_detections)
    det_boxes = torch.cat(det_boxes, dim=0)  # (n_detections, 4)
    det_labels = torch.cat(det_labels, dim=0)  # (n_detections)
    det_scores = torch.cat(det_scores, dim=0)  # (n_detections)


    assert det_images.size(0) == det_boxes.size(0) == det_labels.size(0) == det_scores.size(0)

    # Calculate APs for each class (except background)
    average_precisions = torch.zeros((n_classes - 1), dtype=torch.float)  # (n_classes - 1)
    for c in range(1, n_classes):
        # Extract only objects with this class
        true_class_images = true_images[true_labels == c]  # (n_class_objects)
        true_class_boxes = true_boxes[true_labels == c]  # (n_class_objects, 4)
        true_class_difficulties = true_difficulties[true_labels == c]  # (n_class_objects)
        n_easy_class_objects = (1 - true_class_difficulties).sum().item()  # ignore difficult objects, leftover from older codebase thus we fix the difficulties to zero since we have no "difficult" objects"

        # Keep track of which true objects with this class have already been 'detected'
        # So far, none
        true_class_boxes_detected = torch.zeros((true_class_difficulties.size(0)), dtype=torch.uint8).to(
            device)  # (n_class_objects)

        # Extract only detections with this class
        det_class_images = det_images[det_labels == c]  # (n_class_detections)
        det_class_boxes = det_boxes[det_labels == c]  # (n_class_detections, 4)
        det_class_scores = det_scores[det_labels == c]  # (n_class_detections)
        n_class_detections = det_class_boxes.size(0)
        if n_class_detections == 0:
            continue

        # Sort detections in decreasing order of confidence/scores
        det_class_scores, sort_ind = torch.sort(det_class_scores, dim=0, descending=True)  # (n_class_detections)
        det_class_images = det_class_images[sort_ind]  # (n_class_detections)
        det_class_boxes = det_class_boxes[sort_ind]  # (n_class_detections, 4)

        # In the order of decreasing scores, check if true or false positive
        true_positives = torch.zeros((n_class_detections), dtype=torch.float).to(device)  # (n_class_detections)
        false_positives = torch.zeros((n_class_detections), dtype=torch.float).to(device)  # (n_class_detections)

        for d in range(n_class_detections):
            this_detection_box = det_class_boxes[d].unsqueeze(0)  # (1, 4)
            this_image = det_class_images[d]  # (), scalar

            # Find objects in the same image with this class, their difficulties, and whether they have been detected before
            object_boxes = true_class_boxes[true_class_images == this_image]  # (n_class_objects_in_img)
            object_difficulties = true_class_difficulties[true_class_images == this_image]  # (n_class_objects_in_img)
            # If no such object in this image, then the detection is a false positive
            if object_boxes.size(0) == 0:
                false_positives[d] = 1
                continue

            # Find maximum overlap of this detection with objects in this image of this class
            overlaps = iou(this_detection_box, object_boxes)  # (1, n_class_objects_in_img)

            max_overlap, ind = torch.max(overlaps.squeeze(0), dim=0)  # (), () - scalars

            # 'ind' is the index of the object in these image-level tensors 'object_boxes', 'object_difficulties'
            # In the original class-level tensors 'true_class_boxes', etc., 'ind' corresponds to object with index...
            original_ind = torch.tensor(range(true_class_boxes.size(0)), dtype=torch.long, device=device)[true_class_images == this_image][ind]
            # We need 'original_ind' to update 'true_class_boxes_detected'

            # If the maximum overlap is greater than the threshold of 0.5, it's a match
            if max_overlap.item() > T:
                # If the object it matched with is 'difficult', ignore it
                if object_difficulties[ind] == 0:
                    # If this object has already not been detected, it's a true positive
                    if true_class_boxes_detected[original_ind] == 0:
                        true_positives[d] = 1
                        true_class_boxes_detected[original_ind] = 1  # this object has now been detected/accounted for
                    # Otherwise, it's a false positive (since this object is already accounted for)
                    else:
                        false_positives[d] = 1
            # Otherwise, the detection occurs in a different location than the actual object, and is a false positive
            else:
                false_positives[d] = 1

        # Compute cumulative precision and recall at each detection in the order of decreasing scores
        cumul_true_positives = torch.cumsum(true_positives, dim=0)  # (n_class_detections)
        cumul_false_positives = torch.cumsum(false_positives, dim=0)  # (n_class_detections)
        cumul_precision = cumul_true_positives / (
                cumul_true_positives + cumul_false_positives + 1e-10)  # (n_class_detections)
        cumul_recall = cumul_true_positives / n_easy_class_objects  # (n_class_detections)

        # Find the mean of the maximum of the precisions corresponding to recalls above the threshold 't'
        recall_thresholds = torch.arange(start=0, end=1.1, step=.1).tolist()  # (11)
        precisions = torch.zeros((len(recall_thresholds)), dtype=torch.float).to(device)  # (11)
        for i, t in enumerate(recall_thresholds):
            recalls_above_t = cumul_recall >= t
            if recalls_above_t.any():
                precisions[i] = cumul_precision[recalls_above_t].max()
            else:
                precisions[i] = 0.
        average_precisions[c - 1] = precisions.mean()  # c is in [1, n_classes - 1]

    # Calculate mean of Average Precisions at IoU threshold T, i.e. AP_T
    overall_average_precision = average_precisions.mean().item()

    # Keep class-wise average precisions in a dictionary
    #average_precisions = {rev_label_map[c + 1]: v for c, v in enumerate(average_precisions.tolist())}

    return average_precisions, overall_average_precision
