import paddle
print(paddle.device.is_compiled_with_cuda())  # doit afficher True
paddle.utils.run_check()