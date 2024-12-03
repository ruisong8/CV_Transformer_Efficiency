import torch
from thop import profile
# from archs.ViT_model import get_vit, ViT_Aes
from torchvision.models import resnet50
from fvcore.nn import FlopCountAnalysis, parameter_count
from ptflops import get_model_complexity_info
from calflops import calculate_flops, calculate_flops_hf
from timm.models.vision_transformer import vit_tiny_patch16_224, vit_small_patch16_224
from timm.models.swin_transformer import swin_tiny_patch4_window7_224


methods = ["thop", "fvcore", "ptflops", "calflops"]  # choose any from "thop", "fvcore", "ptflops"
models = [resnet50(), vit_small_patch16_224(), swin_tiny_patch4_window7_224()]

def cal_thop(model, input_tensor):
    print("==========thop==========")
    flops, params = profile(model, inputs=(input_tensor, ))
    print('FLOPs = ' + str(flops/1000**3) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')
    print("==========end==========")
    
def cal_fvcore(model, input_tensor):
    print("==========fvcore==========")
    flops = FlopCountAnalysis(model, input_tensor)
    print("FLOPs: ", str(flops.total()/1000**3) + 'G')
    print("Params: ", str(parameter_count(model)['']/1000**2) + 'M')
    print("==========end==========")
    
def cal_ptflops(model, input_tensor):
    print("==========ptflops==========")
    flops, params = get_model_complexity_info(model, tuple(input_tensor.size()[1:]), as_strings=True, print_per_layer_stat=False)
    print('Flops:  ' + flops)
    print('Params: ' + params)
    print("==========end==========")
    
def cal_calflops(model, input_tensor):
    print("==========calflops==========")
    flops, macs, params = calculate_flops(model=model, 
                                      input_shape=tuple(input_tensor.size()),
                                      output_as_string=True,
                                      print_results=False,
                                      print_detailed=False,
                                      output_precision=4)
    print('macs: ' + macs)
    print('Flops:  ' + flops)
    print('Params: ' + params)
    print("==========end==========")


if __name__ == "__main__":
    input_tensor = torch.randn(1, 3, 224, 224)
    for model in models:
        if "thop" in methods:
            cal_thop(model, input_tensor) 
        if "fvcore" in methods:
            cal_fvcore(model, input_tensor) 
        if "ptflops" in methods:
            cal_ptflops(model, input_tensor)  
        if "calflops" in methods:
            cal_calflops(model, input_tensor)
            
#     batch_size, max_seq_length = 1, 128
#     model_name = "baichuan-inc/Baichuan-13B-Chat"

#     flops, macs, params = calculate_flops_hf(model_name=model_name, input_shape=(batch_size, max_seq_length))
#     print("%s FLOPs:%s  MACs:%s  Params:%s \n" %(model_name, flops, macs, params))
