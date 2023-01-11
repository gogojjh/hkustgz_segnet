import nvidia_smi


def print_gpu_util():

    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    # card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate

    res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
    print(f'gpu: {res.gpu}%, gpu-mem: {res.memory}%')

