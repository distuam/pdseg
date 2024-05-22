import torch
def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def adjust_learning_rate(optimizer, epoch, start_lr, evaluate):
    # Calculate the number of times the learning rate needs to be decreased by 10%
    num_decays = epoch // evaluate
    
    # Calculate the new learning rate after decay
    lr = start_lr * (0.95 ** num_decays)
    
    # Update the learning rate for all parameter groups in the optimizer
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        print(f"Learning rate: {param_group['lr']}")
        break

def print_learning_rate(optimizer):

    # Update the learning rate for all parameter groups in the optimizer
    for param_group in optimizer.param_groups:
        print(f"Learning rate: {param_group['lr']}")
        break