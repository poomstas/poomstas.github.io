---
layout: post
title:  "Multi-GPU Training with PyTorch"
date:   2022-03-07
excerpt: "Some notes about Model Parallel, Data Parallel, and Distributed Data Parallel."
image: "/images/RL_Model_Policy.jpg"
published: false

---

Notes from a great tutorial: https://www.youtube.com/watch?v=TibQO_xv1zc



![image-20220307224923315](../images/20220307_MultiGPU_PyTorch/image-20220307224923315.png)

![image-20220307225013171](../images/20220307_MultiGPU_PyTorch/image-20220307225013171.png)

![image-20220307225033439](../images/20220307_MultiGPU_PyTorch/image-20220307225033439.png)

![image-20220307225054540](../images/20220307_MultiGPU_PyTorch/image-20220307225054540.png)

```python
class ModelParallel(nn.Module):
    def __init__(self, ...):
        super(ModelParallel, self).__init__()
        self.part_1 = nn.Sequential(
            ...
        )
        self.part_2 = nn.Sequential(
            ...
        )
        # Put each part on a different device
        self.part_1.to(torch.device('cuda:0'))
        self.part_2.to(torch.device('cuda:1'))

    def forward(self, x):
        x = x.to(torch.device('cuda:0'))
        x1 = self.part_1(x)
        # Move to second device
        x1 = x1.to(torch.device('cuda:1'))
        y = self.part_2(x1)
        return y
```





![image-20220307225603283](../images/20220307_MultiGPU_PyTorch/image-20220307225603283.png)



## Gradient Accumulation

```python
# Regular Training
for x, y_gt in data:
    opt.zero_grad()
    y_pred = model(x)
    loss = criterion(y, y_gt)
    loss.backward()
    opt.step()
```

```python
# Gradient Accumulation
for x, y_gt in data:
    opt.zero_grad()
    for sub_x, sub_y_gt, in split(x, y_gt):
        sub_y_pred = model(sub_x)
        loss = criterion(sub_y_pred, sub_y_gt)
        loss.backward()
    opt.step()
```



![image-20220308193325665](../images/20220307_MultiGPU_PyTorch/image-20220308193325665.png)



# Data Parallel

## Bottleneck is GPU compute

![image-20220308193414499](../images/20220307_MultiGPU_PyTorch/image-20220308193414499.png)

![image-20220308193439673](../images/20220307_MultiGPU_PyTorch/image-20220308193439673.png)

![image-20220308193516791](../images/20220307_MultiGPU_PyTorch/image-20220308193516791.png)

![image-20220308193632353](../images/20220307_MultiGPU_PyTorch/image-20220308193632353.png)

```python
model = MyModel()
model = nn.DataParallel(model) # make it parallel
```

![image-20220308193815674](/Users/Brian/Dropbox/SideProjects/Blog/poomstas.github.io/images/20220307_MultiGPU_PyTorch/image-20220308193815674.png)

```python
def train(rank, args):
    # init the process in context
    torch.distributed.init_process_group(backend='nccl', 
                                         init_method='tcp://127.0.0.1:54263', 
                                         world_size, rank)
    ...
    # wrap the model
    model = nn.parallel.DistributedDataParallel(mode, device_ids=[rank])

    # use "distributed aware" sampler
    sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank
    )

    loader = DataLoader(train_dataset, batch_size,
                        shuffle=False, sampler=train_sampler)
```

![image-20220308194042332](/Users/Brian/Dropbox/SideProjects/Blog/poomstas.github.io/images/20220307_MultiGPU_PyTorch/image-20220308194042332.png)

![image-20220308194140851](/Users/Brian/Dropbox/SideProjects/Blog/poomstas.github.io/images/20220307_MultiGPU_PyTorch/image-20220308194140851.png)





![image-20220308194226901](/Users/Brian/Dropbox/SideProjects/Blog/poomstas.github.io/images/20220307_MultiGPU_PyTorch/image-20220308194226901.png)





------------------

![image-20220308194332508](/Users/Brian/Dropbox/SideProjects/Blog/poomstas.github.io/images/20220307_MultiGPU_PyTorch/image-20220308194332508.png)

![image-20220308194400685](/Users/Brian/Dropbox/SideProjects/Blog/poomstas.github.io/images/20220307_MultiGPU_PyTorch/image-20220308194400685.png)

![image-20220308194429017](/Users/Brian/Dropbox/SideProjects/Blog/poomstas.github.io/images/20220307_MultiGPU_PyTorch/image-20220308194429017.png)

![image-20220308194526907](/Users/Brian/Dropbox/SideProjects/Blog/poomstas.github.io/images/20220307_MultiGPU_PyTorch/image-20220308194526907.png)

![image-20220308194557262](/Users/Brian/Dropbox/SideProjects/Blog/poomstas.github.io/images/20220307_MultiGPU_PyTorch/image-20220308194557262.png)

![image-20220308194622080](/Users/Brian/Dropbox/SideProjects/Blog/poomstas.github.io/images/20220307_MultiGPU_PyTorch/image-20220308194622080.png)

![image-20220308194641659](/Users/Brian/Library/Application Support/typora-user-images/image-20220308194641659.png)
