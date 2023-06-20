import torch
from model import SimpleRCNN
from main import get_optimizer
from data import get_waste
from mAP import mAP
from tqdm import tqdm
import numpy as np
from utils import label_proposals
import torch.nn as nn
import torch.multiprocessing as mp
import torch.nn.functional as F
import torchvision.transforms as transforms
from selectivesearch import SelectiveSearch
import argparse
from box_ops import nms

def set_args_prediction():
    parser = argparse.ArgumentParser(description="Object Detection Prediction Script")
    parser.add_argument("--model_name", type=str, default='model.pt', help="Write your model name (with .pt)")
    return parser.parse_args()

def predict(model, 
        optimizer,
        test_loader,
        in_batch_size=32,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        ):
    model.to(device)

    resize = transforms.Resize((256, 256), antialias=True)

    # Selective search module
    ss = SelectiveSearch(mode='f', nkeep=400)

    # Criterion for classification loss
    criterion =  nn.CrossEntropyLoss()
    
    pool = mp.Pool(mp.cpu_count())

    # Validation phase
    model.eval()
    test_losses = []
    print_loss_total = 0  # Reset every print_every
    val_correct = 0
    print_val_correct = 0
    N_val = 0
    image_index = 0
    with torch.no_grad():
        pred_boxes_dicts = []
        target_boxes_dicts = []
        for ims, targets in tqdm(test_loader):
            
            
            # Get object proposals for all images in the batch
            proposals_batch = pool.map(ss, [(np.moveaxis(im.numpy(), 0, 2) * 255).astype(np.uint8) for im in ims]) # Multiprocessing for Selective search

            # Get labels and subsample the region proposals for training purposes
            proposals_batch, proposals_batch_labels = label_proposals(proposals_batch, targets, filter=False)
            boxes_batch = proposals_batch
            
            # Labels
            y_true = torch.tensor(np.concatenate(proposals_batch_labels))
            
            # Crop out proposals and resize.
            X = []
            im_indices = []
            valid = []
            idx = 0
            
            for im_idx, (im, boxes) in enumerate(zip(ims, boxes_batch)):
                boxes_ = []
                for (x, y, w, h) in boxes:
                    candidate = im[:, y:y+max(h, 2), x:x+max(w, 2)]
                    if any(torch.tensor(candidate.size()) == 0):
                        idx += 1
                        continue
                    boxes_.append(torch.tensor([x,y,w,h]))
                    X.append(resize.forward(candidate))
                    im_indices.append(im_idx)
                    valid.append(idx)
                    idx += 1
                pred_boxes_dicts.append({'boxes':torch.stack(boxes_)})
            X = torch.stack(X)
            y_true = y_true[torch.tensor(valid)]

            #We don't shuffle here! :D
            shuffle = torch.arange(y_true.size(0))

            pred_labels = []
            conf_labels = []

            
            for j in range(shuffle.size(0) // in_batch_size + bool(shuffle.size(0) % in_batch_size)):
                indices = shuffle[j * in_batch_size: (j + 1) * in_batch_size]
                X_batch, y_batch_true = X[indices], y_true[indices]
                
                X_batch = X_batch.to(device)
                y_batch_true = y_batch_true.to(device)
                y_batch_true = y_batch_true.long()
                
                output = model(X_batch)
                
                loss = criterion(output, y_batch_true)
                
                with torch.no_grad():
                    p = F.softmax(output, dim=1).max(1)
                    preds = p.indices
                    conf = p.values
                    
                    #Do NMS!
                    #pred_boxes_dicts_pre_nms[im_idx]['boxes']

                    pred_labels.append(preds)
                    conf_labels.append(conf)
                    n_correct = (y_batch_true == preds).sum().cpu().item()
                    val_correct += n_correct
                    print_val_correct += n_correct
                    test_losses.append(loss.item())


                print_loss_total += loss.item()
                
            
            pred_labels_ = torch.hstack(pred_labels)
            conf_labels_ = torch.hstack(conf_labels)
            im_indices_ = torch.tensor(im_indices)
            
            for i in range(len(ims)):
                id = i+image_index
                labels = pred_labels_[im_indices_==i].cpu()
                confs = conf_labels_[im_indices_==i].cpu()
                labels_unique = torch.unique(labels)
                kept_boxes = []
                kept_scores = []
                kept_labels = []
                for label in labels_unique:
                    if label == 0:
                        continue

                    indices_ = torch.argwhere(labels==label).squeeze(1)
                    
                    picked_boxes, picked_scores = nms(pred_boxes_dicts[id]['boxes'][indices_,:].numpy(), scores=confs[indices_].numpy())
                    kept_boxes.append(torch.tensor(picked_boxes))
                    kept_scores.append(torch.tensor(picked_scores))
                    kept_labels.append(torch.zeros(len(indices_))+label)
                kept_boxes = torch.vstack(kept_boxes)
                kept_scores = torch.hstack(kept_scores)
                kept_labels = torch.hstack(kept_labels)

                print("boxes.shape: ",kept_boxes.shape)
                print()
                print("scores.shape: ", kept_scores.shape)
                print()
                print("labels.shape: ", kept_labels.shape)

                pred_boxes_dicts[id]['labels'] = kept_labels
                pred_boxes_dicts[id]['scores'] = kept_scores
                pred_boxes_dicts[id]['boxes'] = kept_boxes
                target_boxes_dicts.append({'boxes': targets[i]['bboxes'], 'labels': targets[i]['category_ids']})

            
            image_index += len(ims)
            N_batch = y_true.size(0)
            N_val += N_batch
            print_loss_avg = print_loss_total / (j + 1)
            print(f"Average loss: {print_loss_avg:.2f}")
            print(f"Average accuracy: {print_val_correct / N_batch:.3%}")
            print_loss_total = 0
            print_val_correct = 0
        
    return pred_boxes_dicts, target_boxes_dicts
    
    
    

def main():
    args_pred = set_args_prediction()

    print(args_pred)
    checkpoint = torch.load(f'models/{args_pred.model_name}')
    args = vars(checkpoint['args'])

    model = SimpleRCNN(args['n_layers'], args['n_classes'])
    optimizer = get_optimizer(args['optimizer_type'], model, args['pretrained_lr'], args['new_layer_lr'])
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    #Load test data
    _, _, _, train_loader, val_loader, test_loader = get_waste(32, #args['batch_size'],
              num_workers=8,
              data_augmentation=args['data_augmentation'],
              supercategories=True)
    
    preds, targets = predict(
        model=model,
        optimizer=optimizer,
        test_loader=test_loader,
        in_batch_size= 32
    )
    print(mAP(preds, targets))

    x=1




if __name__=='__main__':
    main()
    print('done')

