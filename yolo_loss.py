import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def compute_iou(box1, box2):
    """Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
    Args:
      box1: (tensor) bounding boxes, sized [N,4].
      box2: (tensor) bounding boxes, sized [M,4].
    Return:
      (tensor) iou, sized [N,M].
    """
    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(
        box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    rb = torch.min(
        box1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, 2:].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    wh = rb - lt  # [N,M,2]
    wh[wh < 0] = 0  # clip at 0
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
    area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
    area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

    iou = inter / (area1 + area2 - inter)
    return iou


class YoloLoss(nn.Module):
    def __init__(self, S, B, l_coord, l_noobj):
        super(YoloLoss, self).__init__()
        self.S = S
        self.B = B
        self.l_coord = l_coord
        self.l_noobj = l_noobj

    def xywh2xyxy(self, boxes):
        """
        Parameters:
        boxes: (N,4) representing by x,y,w,h

        Returns:
        boxes: (N,4) representing by x1,y1,x2,y2

        if for a Box b the coordinates are represented by [x, y, w, h] then
        x1, y1 = x/S - 0.5*w, y/S - 0.5*h ; x2,y2 = x/S + 0.5*w, y/S + 0.5*h
        Note: Over here initially x, y are the center of the box and w,h are width and height.
        """
        boxes[:,0], boxes[:, 1] = boxes[:,0]/self.S - 0.5*boxes[:,2], boxes[:,1]/self.S - 0.5*boxes[:,3]
        boxes[:,2], boxes[:, 3] = boxes[:,0]/self.S + 0.5*boxes[:,2], boxes[:,1]/self.S + 0.5*boxes[:,3]
        ### CODE ###
        # Your code here

        return boxes

    def find_best_iou_boxes(self, pred_box_list, box_target):
        """
        Parameters:
        box_pred_list : [(tensor) size (-1, 5) ...]
        box_target : (tensor)  size (-1, 4)

        Returns:
        best_iou: (tensor) size (-1, 1)
        best_boxes : (tensor) size (-1, 5), containing the boxes which give the best iou among the two (self.B) predictions

        
        """

        
        
        box_target = self.xywh2xyxy(box_target)
        pred_box_list1 = [self.xywh2xyxy(pred_box_list[i][..., :4]) for i in range(self.B)]
        
        iou1 = torch.diagonal(compute_iou(pred_box_list1[0], box_target))#N, M
        iou2 = torch.diagonal(compute_iou(pred_box_list1[1], box_target))#N, M
        
        mask = iou1 > iou2
        
        best_ious = torch.where(mask, iou1, iou2)
        best_boxes = torch.where(mask.unsqueeze(-1).expand_as(pred_box_list[0]), pred_box_list[0], pred_box_list[1])
        
        #return iou1, iou2
        return best_ious, best_boxes

    def get_class_prediction_loss(self, classes_pred, classes_target, has_object_map):
        """
        Parameters:
        classes_pred : (tensor) size (batch_size, S, S, 20)
        classes_target : (tensor) size (batch_size, S, S, 20)
        has_object_map: (tensor) size (batch_size, S, S)

        Returns:
        class_loss : scalar
        """
        
        N = classes_pred.size(0)
        #has_map = has_object_map#.view(N, self.S, self.S, -1)
        loss = torch.sum(has_object_map * torch.sum(torch.pow((classes_target - classes_pred), 2), axis = -1))
        return loss

    def get_no_object_loss(self, pred_boxes_list, has_object_map):
        """
        Parameters:
        pred_boxes_list: (list) [(tensor) size (N, S, S, 5)  for B pred_boxes]
        has_object_map: (tensor) size (N, S, S)

        Returns:
        loss : scalar
        """
        ### CODE ###
        # Your code here
        has_object_rev = has_object_map.clone()
        has_object_rev = ~has_object_rev
        has_object_rev = has_object_rev.long()
        
        
        loss = 0.0
        for i in range(self.B):
            loss += has_object_rev * torch.pow(-pred_boxes_list[i][:, :, :, -1], 2)#N, S, S
        loss = torch.sum(loss) * self.l_noobj
        
        return loss

    def get_contain_conf_loss(self, box_pred_conf, box_target_conf):
        """
        Parameters:
        box_pred_conf : (tensor) size (-1,1)
        box_target_conf: (tensor) size (-1,1)

        Returns:
        contain_loss : scalar


        """
        ### CODE
        # your code here
        loss = 0.0
        loss += torch.pow((box_target_conf.detach() - box_pred_conf), 2)
        loss = torch.sum(loss)
        
        return loss

    def get_regression_loss(self, box_pred_response, box_target_response):
        """
        Parameters:
        box_pred_response : (tensor) size (-1, 4)
        box_target_response : (tensor) size (-1, 4)
        Note : -1 corresponds to ravels the tensor into the dimension specified
        See : https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view_as

        Returns:
        reg_loss : scalar

        """
        ### CODE
        # your code here
        centers_pred = box_pred_response[..., :2]
        centers_target = box_target_response[..., :2]
        
        width_height_pred = box_pred_response[..., 2:]
        width_height_target = box_target_response[..., 2:]
        
        loss = torch.sum(torch.pow(centers_pred - centers_target, 2))
        loss += torch.sum(
            torch.pow(torch.sqrt(width_height_pred) - torch.sqrt(width_height_target), 2)
        )
        
        
        
        return self.l_coord * loss

    def forward(self, pred_tensor, target_boxes, target_cls, has_object_map):
        """
        pred_tensor: (tensor) size(N,S,S,Bx5+20=30) N:batch_size
                      where B - number of bounding boxes this grid cell is a part of = 2
                            5 - number of bounding box values corresponding to [x, y, w, h, c]
                                where x - x_coord, y - y_coord, w - width, h - height, c - confidence of having an object
                            20 - number of classes

        target_boxes: (tensor) size (N, S, S, 4): the ground truth bounding boxes
        target_cls: (tensor) size (N, S, S, 20): the ground truth class
        has_object_map: (tensor, bool) size (N, S, S): the ground truth for whether each cell contains an object (True/False)

        Returns:
        loss_dict (dict): with key value stored for total_loss, reg_loss, containing_obj_loss, no_obj_loss and cls_loss
        """
        N = pred_tensor.size(0)
        total_loss = 0.0

        # split the pred tensor from an entity to separate tensors:
        # -- pred_boxes_list: a list containing all bbox prediction (list) [(tensor) size (N, S, S, 5)  for B pred_boxes]
        # -- pred_cls (containing all classification prediction)
        pred_boxes_list = [pred_tensor[:, :, :, 5*i:5 + 5*i] for i in range(self.B)]
        pred_cls = pred_tensor[:, :, :, 10:30]

        # compcute classification loss
        classification_loss = self.get_class_prediction_loss(pred_cls, target_cls, has_object_map) * 1/N
        # compute no-object loss
        no_object_loss =  self.get_no_object_loss(pred_boxes_list, has_object_map) * 1/N
        # Re-shape boxes in pred_boxes_list and target_boxes to meet the following desires
        # 1) only keep having-object cells
        # 2) vectorize all dimensions except for the last one for faster computation
        '''
        better approach - decompose the first dimension for both tensors but going with my first intuitive solution
        box1 = pred_boxes_list[0]
        box1 = box1.view(-1, 5)#N*S*S, 5
        box1[has_object_map.view(-1), :]#has_object_map.view(-1)->N*S*S(1 dimension tensor)
        '''
        has_map_reshape = has_object_map.view(N, self.S, self.S, 1).long()
        broadcast_tensor_box1 = has_map_reshape * pred_boxes_list[0]
        box1 = broadcast_tensor_box1[~torch.all(broadcast_tensor_box1==0, axis=3)] #(-1, 5)
        broadcast_tensor_box2 = has_map_reshape * pred_boxes_list[1]
        box2 = broadcast_tensor_box2[~torch.all(broadcast_tensor_box2==0, axis=3)]#(-1, 5) 
        pred_boxes_list = [box1, box2]
        
        target_boxes = target_boxes.reshape(-1, 4)[has_object_map.view(-1), :]
        # find the best boxes among the 2 (or self.B) predicted boxes and the corresponding iou
        best_ious, best_boxes = self.find_best_iou_boxes(pred_boxes_list, target_boxes)
        # compute regression loss between the found best bbox and GT bbox for all the cell containing objects
        reg_loss = self.get_regression_loss(best_boxes[..., :-1], target_boxes) * 1/N
        # compute contain_object_loss
        contain_object_loss = self.get_contain_conf_loss(best_boxes[..., -1], best_ious) * 1/N
        # compute final loss
        final_loss = (classification_loss + no_object_loss + reg_loss + contain_object_loss)
        # construct return loss_dict
        loss_dict = dict(
            total_loss=final_loss,
            reg_loss=reg_loss,
            containing_obj_loss=contain_object_loss,
            no_obj_loss=no_object_loss,
            cls_loss=classification_loss,
        )
        return loss_dict
        #return best_ious, best_boxes, pred_boxes_list
