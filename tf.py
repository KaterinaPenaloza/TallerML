import tensorflow as tf

from tensorflow.python.client import device_lib
#device_lib.list_local_devices()

#tf.test.is_built_with_cuda(), tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)

# Verifica los dispositivos disponibles (debería mostrar tu GPU)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

#with tf.device('/cpu:0'):

# import tensorflow as tf

# # Verificar GPUs disponibles
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         # Configurar TensorFlow para usar GPU
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         print("GPUs disponibles:", len(gpus))
#     except RuntimeError as e:
#         print(e)
# else:
#     print("No se encontraron GPUs disponibles.")