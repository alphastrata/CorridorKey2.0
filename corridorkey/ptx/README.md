# PTX (parallel thread execution)

NVIDIA's 'portable assembly' format, you can ship this then using `nvcc` via the vanilla torch tooling etc compile these to run on NVIDIA's gpus.

This whole module is just to make that easier.

Important environment variables this module is aware of:
`CORRIDORKEY_PTX` if set to `1`, it is ON!, you _CAN_ set it to 0, which will disable it, and then you'll be back on `torch`'s unoptimised kernels.