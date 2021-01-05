import torch
import sys
sys.path.append('../')
from context_rcnn.models.context_rcnn import Context_RCNN

def get_dummy_inputs():
    images = torch.rand((2,3,720,1280))
    boxes, labels = torch.rand((2,10,4)), torch.randint(1,3,(2,10))
    images_ = list(image for image in images)
    targets = []
    for i in range(len(images_)):
        d = {}
        d['boxes'] = boxes[i]
        d['labels'] = labels[i]
        targets.append(d)
    
    images2 = torch.rand((2,2,3,720,1280))
    boxes, labels = torch.rand((2,2,10,4)), torch.randint(1,3,(2,2,10))
    targets2 = []
    for i,img in enumerate(images2):
        images_ = list(image for image in img)
        targets = []
        for j in range(len(images_)):
            d = {}
            d['boxes'] = boxes[i][j]
            d['labels'] = labels[i][j]
            targets.append(d)
        targets2.append(targets)
    
    return images, targets, images2, targets2

def test_context_rcnn_predict():
    img1, targets1, imgs2, targets2 = get_dummy_inputs()

    img_batch1 = img1.cuda().float()
    img_batch2 = imgs2.cuda().float()
    targets1 = [{k: v.cuda() for k, v in t.items()} for t in targets1]
    targets3 = []
    for tar in targets2:
      targets2_ = [{k: v.cuda() for k, v in t.items()} for t in tar]
      targets3.append(targets2_)
    targets2 = targets3

    model = Context_RCNN('resnet50', num_classes=3, use_long_term_attention=True,
                      backbone_out_features=256, attention_features=2048,
                      attention_post_rpn=True, attention_post_box_classifier=False, 
                      use_self_attention=False, self_attention_in_sequence=False, 
                      num_attention_heads=1, num_attention_layers=1)
    model.cuda()

    model.train()
    dets, loss_dict = model(img_batch1, img_batch2, targets1, context_targets=targets2)
    print(loss_dict)
    
    model.eval()
    dets, loss_dict = model(img_batch1, img_batch2)

    assert dets[0]['boxes'][0].shape == targets1[0]['boxes'][0].shape
    assert dets[0]['labels'][0].shape == targets1[0]['labels'][0].shape
    assert dets[0]['scores'][0].shape == targets1[0]['labels'][0].shape
    
if __name__ == '__main__':
    test_context_rcnn_predict()
