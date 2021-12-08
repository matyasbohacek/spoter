
import logging
import torch


def train_epoch(model, dataloader, criterion, optimizer, device, scheduler=None):

    pred_correct, pred_all = 0, 0
    running_loss = 0.0

    for i, data in enumerate(dataloader):
        inputs, labels = data
        inputs = inputs.squeeze(0).to(device)
        labels = labels.to(device, dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(inputs).expand(1, -1, -1)

        loss = criterion(outputs[0], labels[0])
        loss.backward()
        optimizer.step()
        running_loss += loss

        # Statistics
        if int(torch.argmax(torch.nn.functional.softmax(outputs, dim=2))) == int(labels[0][0]):
            pred_correct += 1
        pred_all += 1

    if scheduler:
        scheduler.step(running_loss.item() / len(dataloader))

    return running_loss, pred_correct, pred_all, (pred_correct / pred_all)


def evaluate(model, dataloader, device, print_stats=False):

    pred_correct, pred_all = 0, 0
    stats = {i: [0, 0] for i in range(101)}

    for i, data in enumerate(dataloader):
        inputs, labels = data
        inputs = inputs.squeeze(0).to(device)
        labels = labels.to(device, dtype=torch.long)

        outputs = model(inputs).expand(1, -1, -1)

        # Statistics
        if int(torch.argmax(torch.nn.functional.softmax(outputs, dim=2))) == int(labels[0][0]):
            stats[int(labels[0][0])][0] += 1
            pred_correct += 1

        stats[int(labels[0][0])][1] += 1
        pred_all += 1

    if print_stats:
        stats = {key: value[0] / value[1] for key, value in stats.items() if value[1] != 0}
        print("Label accuracies statistics:")
        print(str(stats) + "\n")
        logging.info("Label accuracies statistics:")
        logging.info(str(stats) + "\n")

    return pred_correct, pred_all, (pred_correct / pred_all)


def evaluate_top_k(model, dataloader, device, k=5):

    pred_correct, pred_all = 0, 0

    for i, data in enumerate(dataloader):
        inputs, labels = data
        inputs = inputs.squeeze(0).to(device)
        labels = labels.to(device, dtype=torch.long)

        outputs = model(inputs).expand(1, -1, -1)

        if int(labels[0][0]) in torch.topk(outputs, k).indices.tolist():
            pred_correct += 1

        pred_all += 1

    return pred_correct, pred_all, (pred_correct / pred_all)
