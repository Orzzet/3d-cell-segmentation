Prerequisitos:
+ CUDA https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
+ CUDNN https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html
+ Python 3.5+ https://www.python.org/downloads/
+ PyTorch https://github.com/pytorch/pytorch
+ Jupyter Notebook https://jupyter.readthedocs.io/en/latest/install.html} o un Software que pueda ejectur Notebooks

Cómo usarlo:
1. Descargar el proyecto del repositorio y descomprimir en el lugar deseado.
2. El notebook 1\_procesado es para hacer el entrenamiento.
3. El notebook 2\_inferencia es para utilizar el modelo, bien para estadísticas o para comprobar el resultado.

Cómo añadir datasets:
1. Se deberán subir los archivos en la carpeta \data
2. El notebook 0\_preprocesado trata las imágenes y las transforma en el formato correcto. Modificar este archivo en función del tipo de dato subido.
3. Modificar el archivo dataset.py y añadir el PATH a los datasets nuevos.
