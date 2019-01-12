from utils.flops_count import *
from torch.utils import data
from dataset import get_segmentation_dataset
from network import get_segmentation_model
from config import Parameters
import numpy as np
from torch.autograd import Variable
import timeit
args = Parameters().parse()
args.batch_size = 1
args.dataset = 'cityscapes_light'

methods=['student_res18_pre']
args.data_list='/teamscratch/msravcshare/v-yifan/deeplab_v3/dataset/list/cityscapes/val.lst'

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
testloader = data.DataLoader(get_segmentation_dataset(args.dataset, root=args.data_dir, list_path=args.data_list,
                                                      crop_size=(1024, 2048), mean=IMG_MEAN, scale=False,
                                                      mirror=False),
                             batch_size=args.batch_size, shuffle=False, pin_memory=True)

for method in methods:
    args.method = method
    student = get_segmentation_model(args.method, num_classes=args.num_classes)
    # from network.md import MobileNet
    # student=MobileNet()
    student = add_flops_counting_methods(student)
    student = student.cuda()
    student = student.eval()

    student.start_flops_count()

    print('method:',method)
    for i_iter, batch in enumerate(testloader):
        i_iter += args.start_iters
        images, labels, _, _ = batch
        images = Variable(images.cuda())
        labels = Variable(labels.long().cuda())
        start = timeit.default_timer()
        preds = student(images)
        end = timeit.default_timer()
        print(end - start, 'seconds')
        if i_iter < 1:
            flops = student.compute_average_flops_cost()/1e9
            print('ok!!',flops/2)
        if i_iter > 5:
            break
