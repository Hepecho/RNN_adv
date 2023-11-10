import torch
import time
from utils import *
from runx.logx import logx
from os.path import join as ospj
import os


def train(model, iterator, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:
        batch_label, batch_text = batch
        batch_text = batch_text.to(device)
        # batch_text = batch_text.permute(1, 0)
        batch_label = batch_label.to(device, dtype=torch.float)

        optimizer.zero_grad()

        predictions = model(batch_text).squeeze(1)

        loss = criterion(predictions, batch_label)

        acc = binary_accuracy(predictions, batch_label)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()
        # print(acc.item())

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            batch_label, batch_text = batch
            batch_text = batch_text.to(device)
            # batch_text = batch_text.permute(1, 0)
            batch_label = batch_label.to(device, dtype=torch.float)
            predictions = model(batch_text).squeeze(1)

            loss = criterion(predictions, batch_label)

            acc = binary_accuracy(predictions, batch_label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def train_model(args, model, optimizer, criterion, train_iterator, valid_iterator, test_iterator, device):
    model = model.to(device)
    criterion = criterion.to(device)

    best_valid_loss = float('inf')
    N_EPOCHS = args.epochs
    best_model_path = ospj(args.ckptdir, args.model, 'best_model.pt')
    last_model_path = ospj(args.ckptdir, args.model, 'last_model.pt')
    localtime = time.asctime(time.localtime(time.time()))
    logx.msg('======================Start Train Model [{}]===================='.format(localtime))

    for epoch in range(N_EPOCHS):

        start_time = time.time()
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion, device)
        logx.add_scalar('train_loss', train_loss, epoch)
        logx.add_scalar('train_acc', train_acc, epoch)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, device)
        logx.add_scalar('valid_loss', valid_loss, epoch)
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), best_model_path)
        if epoch == N_EPOCHS - 1:
            torch.save(model.state_dict(), last_model_path)

        logx.msg('Epoch: {} | Epoch Time: {}m {}s'.format(epoch + 1, epoch_mins, epoch_secs))
        logx.msg('Train Loss: {} | Train Acc: {}%'.format(train_loss, train_acc * 100))
        logx.msg('Val. Loss: {} | Val. Acc: {}%'.format(valid_loss, valid_acc * 100))

    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_acc = evaluate(model, test_iterator, criterion, device)

    # print('Test Loss: {:.3f} | Test Acc: {:.2f}%'.format(test_loss, test_acc * 100))
    logx.msg('Test Loss: {:.3f} | Test Acc: {:.2f}%'.format(test_loss, test_acc * 100))

    localtime = time.asctime(time.localtime(time.time()))
    logx.msg('======================Finish Train Model [{}]===================='.format(localtime))
    return model
