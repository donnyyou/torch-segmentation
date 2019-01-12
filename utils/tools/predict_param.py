import logging

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from collections import OrderedDict


def cost_summary(model, input_size, device="cuda", logfile='param.log'):
        def register_hook(module):
            def hook(module, input, output):
                if (not (hasattr(module, 'weight') and hasattr(module.weight, 'size'))
                    and not (hasattr(module, 'bias') and hasattr(module.bias, 'size'))):
                    return
                class_name = str(module.__class__).split('.')[-1].split("'")[0]
                module_idx = len(summary)

                m_key = '%s-%i' % (class_name, module_idx+1)
                summary[m_key] = OrderedDict()
                summary[m_key]['input_shape'] = list(input[0].size())
                summary[m_key]['input_shape'][0] = -1
                if isinstance(output, (list,tuple)):
                    summary[m_key]['output_shape'] = [[-1] + list(o.size())[1:] for o in output]
                else:
                    summary[m_key]['output_shape'] = list(output.size())
                    summary[m_key]['output_shape'][0] = -1

                params = 0
                madds = 0
                if hasattr(module, 'weight') and hasattr(module.weight, 'size'):
                    params += torch.prod(torch.LongTensor(list(module.weight.size())))
                    summary[m_key]['trainable'] = module.weight.requires_grad
                    if 'Conv' in class_name:
                        madds += params * torch.prod(torch.LongTensor(
                                                summary[m_key]['output_shape'][2:]))
                    elif 'Linear' in class_name:
                        madds += params
                if hasattr(module, 'bias') and hasattr(module.bias, 'size'):
                    params +=  torch.prod(torch.LongTensor(list(module.bias.size())))
                summary[m_key]['nb_params'] = params
                summary[m_key]['madds'] = madds

            if (not isinstance(module, nn.Sequential) and 
               not isinstance(module, nn.ModuleList) and 
               not (module == model)):
                hooks.append(module.register_forward_hook(hook))
        
        device = device.lower()
        assert device in ["cuda", "cpu"], "Input device is not valid, please specify 'cuda' or 'cpu'"

        if device == "cuda" and torch.cuda.is_available():
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor
        
        # check if there are multiple inputs to the network
        if isinstance(input_size[0], (list, tuple)):
            x = [Variable(torch.rand(2,*in_size)).type(dtype) for in_size in input_size]
        else:
            x = Variable(torch.rand(2,*input_size)).type(dtype)
            
            
        # print(type(x[0]))
        # create properties
        summary = OrderedDict()
        hooks = []
        # register hook
        model.apply(register_hook)
        # make a forward pass
        # print(x.shape)
        model(x)
        # remove these hooks
        for h in hooks:
            h.remove()

        logging.basicConfig(filename=logfile, level=logging.INFO)
        logging.info('----------------------------------------------------------------')
        line_new = '{:>20}  {:>25} {:>15} {:>25}'.format('Layer (type)', 'Output Shape', 'Param #', 'MAdds #')
        logging.info(line_new)
        logging.info('================================================================')
        total_params = 0
        total_madds = 0
        trainable_params = 0
        for layer in summary:
            # input_shape, output_shape, trainable, nb_params
            line_new = '{:>20}  {:>25} {:>15} {:>25}'.format(layer, str(summary[layer]['output_shape']), 
                            '{0:,}'.format(summary[layer]['nb_params']), '{0:,}'.format(summary[layer]['madds']))
            total_params += summary[layer]['nb_params']
            total_madds += summary[layer]['madds']
            if 'trainable' in summary[layer]:
                if summary[layer]['trainable'] == True:
                    trainable_params += summary[layer]['nb_params']
            logging.info(line_new)
        logging.info('================================================================')
        logging.info('Total params: {0:,}'.format(total_params))
        logging.info('Trainable params: {0:,}'.format(trainable_params))
        logging.info('Total madds: {0:,}'.format(total_madds))
        logging.info('Non-trainable params: {0:,}'.format(total_params - trainable_params))
        logging.info('----------------------------------------------------------------')
        print('================================================================')
        print('Total params: {0:,}'.format(total_params))
        print('Trainable params: {0:,}'.format(trainable_params))
        print('Total madds: {0:,}'.format(total_madds))
        print('Non-trainable params: {0:,}'.format(total_params - trainable_params))
        print('----------------------------------------------------------------')
        # return summary



from config import Parameters
from network import get_segmentation_model

def main():
    args = Parameters().parse()
    model = get_segmentation_model("_".join([args.network, args.method]), num_classes=20)
    cost_summary(model=model.cuda(), input_size=(3, 1024, 2048))


if __name__ == '__main__':
    main()