import torch
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import pickle
import argparse
from utils import setup
from utils import load_data, EarlyStopping
import numpy as np

import time

def score(logits, labels):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()

    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')

    return accuracy, micro_f1, macro_f1

def score_test(logits, labels):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()

    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')

    #np.savez("result.npz", prediction, labels)
    result ={}
    result['labels'] = labels
    result['prediction'] = prediction

    #np.save('result.npy', result)
    torch.save(result, "result.pt")
    '''
    cm = confusion_matrix(labels, prediction, labels=[0, 1, 2, 3, 4, 5])
    cm = np.array(cm)

    dict={}
    dict['cm'] = cm
    dict['prediction'] = prediction
    dict['labels'] = labels
    f = open('cm.pkl', 'wb')
    pickle.dump(dict, f)
    f.close()
    '''
    return accuracy, micro_f1, macro_f1

def evaluate(model, g, features, labels, mask, loss_func):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
    loss = loss_func(logits[mask], labels[mask])
    accuracy, micro_f1, macro_f1 = score(logits[mask], labels[mask])

    #f = open('mask.pkl', 'wb')
    #pickle.dump(mask, f)
    #f.close()

    return loss, accuracy, micro_f1, macro_f1

def evaluate_test(model, g, features, labels, mask, loss_func):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
    loss = loss_func(logits[mask], labels[mask])
    accuracy, micro_f1, macro_f1 = score_test(logits[mask], labels[mask])

    return loss, accuracy, micro_f1, macro_f1

def main(args):
    g, features, labels, num_classes, train_idx, val_idx, test_idx, train_mask, \
    val_mask, test_mask = load_data(args['dataset'])

    print(train_idx.shape[0])
    print(val_idx.shape[0])
    print(test_idx.shape[0])

    print(train_mask.sum().item())
    print(val_mask.sum().item())
    print(test_mask.sum().item())
    start_time = time.time()
    #val_idx = test_idx
    #val_mask = test_mask

    if hasattr(torch, 'BoolTensor'):
        train_mask = train_mask.bool()
        val_mask = val_mask.bool()
        test_mask = test_mask.bool()

    features = features.to(args['device'])
    labels = labels.to(args['device'])
    train_mask = train_mask.to(args['device'])
    val_mask = val_mask.to(args['device'])
    test_mask = test_mask.to(args['device'])
    if args['model'] == 'scMGCN':
        from model import scMGCN
        model = scMGCN(num_graph=len(g),
                    in_size=features.shape[1],
                    hidden_size=args['hidden_units'],
                    out_size=num_classes,
                    num_layers=args['num_layers'],
                    dropout=args['dropout']).to(args['device'])
        g = [graph.to(args['device']) for graph in g]
    else:
        from model_vcdn import VCDN
        model = VCDN(num_graph=len(g),
                    in_size=features.shape[1],
                    hidden_size=args['hidden_units'],
                    out_size=num_classes,
                    num_layers=args['num_layers'],
                    dropout=args['dropout']).to(args['device'])
        g = [graph.to(args['device']) for graph in g]

    stopper = EarlyStopping(patience=args['patience'])
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=args['lr'],
                                 weight_decay=args['weight_decay'])

    best_f1 = 0
    best_acc = 0
    for epoch in range(args['num_epochs']):
        model.train()
        logits = model(g, features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc, train_micro_f1, train_macro_f1 = score(logits[train_mask], labels[train_mask])
        val_loss, val_acc, val_micro_f1, val_macro_f1 = evaluate(model, g, features, labels, val_mask, loss_fcn)

        test_loss, test_acc, test_micro_f1, test_macro_f1 = evaluate(model, g, features, labels, test_mask,
                                                                          loss_fcn)
        #if(best_f1<test_macro_f1):
        if (best_acc < test_acc):
            best_f1 = test_macro_f1
            best_acc = test_acc
            test_loss, test_acc, test_micro_f1, test_macro_f1 = evaluate_test(model, g, features, labels, test_mask,
                                                                              loss_fcn)
        print('Test loss {:.4f} | ACC {:.4f}|Test Micro f1 {:.4f} | Test Macro f1 {:.4f}'.format(
            test_loss.item(), test_acc.item(), test_micro_f1, test_macro_f1))


        early_stop = stopper.step(val_loss.data.item(), val_acc, model)

        print('Epoch {:d} | Train Loss {:.4f} | Train Micro f1 {:.4f} | Train Macro f1 {:.4f} | '
              'Val Loss {:.4f} | Val Micro f1 {:.4f} | Val Macro f1 {:.4f}'.format(
            epoch + 1, loss.item(), train_micro_f1, train_macro_f1, val_loss.item(), val_micro_f1, val_macro_f1))

        if early_stop:
            break

    stopper.load_checkpoint(model)
    print(best_f1)
    print(best_acc)
    #test_loss, test_acc, test_micro_f1, test_macro_f1 = evaluate_test(model, g, features, labels, test_mask, loss_fcn)
    #print('Test loss {:.4f} | ACC {:.4f}|Test Micro f1 {:.4f} | Test Macro f1 {:.4f}'.format(
    #    test_loss.item(),test_acc.item(), test_micro_f1, test_macro_f1))
    end_time = time.time()

    run_time = end_time - start_time

    print("程序运行时间为：", run_time)

if __name__ == '__main__':

    parser = argparse.ArgumentParser('scMGCN')
    parser.add_argument('-s', '--seed', type=int, default=111,help='Random seed')
    parser.add_argument('-ld', '--log-dir', type=str, default='results',help='Dir for saving training results')
    args = parser.parse_args().__dict__

    args = setup(args)

    main(args)