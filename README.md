# LabJetsonNano
Pruebas varias con dispositivo

# Configuración de NVIDIA Jetson Orin Nano para uso de GPU

Guía práctica orientada a dejar la GPU **operativa, verificable y explotable**.  
Se asume **Jetson Orin Nano con JetPack 5.x o 6.x** sobre Ubuntu L4T.

## 1. Requisitos previos

La GPU no se activa manualmente. Funciona si el stack es correcto.

Necesitas:
- NVIDIA Jetson Orin Nano
- JetPack instalado
- Ubuntu L4T (incluido en JetPack)
- Fuente de alimentación adecuada

Una alimentación insuficiente provoca throttling o desactivación parcial de la GPU.

## 2. Instalación correcta de JetPack

### Opción recomendada
Instalar JetPack mediante **NVIDIA SDK Manager** desde un host Linux x86.

JetPack incluye:
- Drivers GPU
- CUDA
- cuDNN
- TensorRT
- OpenCV con aceleración
- Nsight

Sin JetPack no hay GPU utilizable.

### Verificación de versión
```bash
cat /etc/nv_tegra_release
````

Salida esperada:

```
# R35.x.x  (JetPack 5.x)
# R36.x.x  (JetPack 6.x)
```

## 3. Verificar que la GPU está disponible

```bash
tegrastats
```

Debe aparecer algo similar a:

```
GR3D_FREQ 0%@918
```

Si aparece `GR3D_FREQ`, la GPU está activa.

## 4. Verificar CUDA

```bash
nvcc --version
```

Si `nvcc` no existe, CUDA no está instalado o no está en el PATH.

Verificación directa:

```bash
ls /usr/local/cuda
```

## 5. Forzar modo de máximo rendimiento

Por defecto la Orin Nano limita clocks.

### Consultar modos

```bash
sudo nvpmodel -q
```

### Activar modo máximo

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

Efecto:

* CPU a máximo rendimiento
* GPU a máximo rendimiento
* Sin escalado dinámico

Obligatorio para benchmarks y pruebas serias.

## 6. Prueba mínima de CUDA en Python

### PyTorch con GPU

Jetson **no usa wheels estándar de PyTorch**.

Ejemplo para JetPack 5.x:

```bash
pip install torch torchvision \
  --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v512
```

Verificación:

```python
import torch
torch.cuda.is_available()
torch.cuda.get_device_name(0)
```

Salida esperada:

```
True
'Orin'
```

## 7. OpenCV con CUDA

```bash
python3 - << EOF
import cv2
print(cv2.getBuildInformation())
EOF
```

Buscar:

* `CUDA: YES`
* `cuDNN: YES`

Si no aparece, OpenCV no está acelerado.

## 8. TensorRT

Imprescindible para inferencia eficiente.

Verificación:

```bash
dpkg -l | grep tensorrt
```

Prueba rápida:

```bash
trtexec --help
```

TensorRT es el componente que realmente exprime la GPU.

## 9. Uso de GPU en Docker

Jetson no usa `nvidia-docker` clásico. El runtime viene integrado.

Verificación:

```bash
docker info | grep -i nvidia
```

Ejemplo de ejecución:

```bash
docker run --rm --runtime nvidia \
  nvcr.io/nvidia/l4t-base:r35.4.1 nvidia-smi
```

Nota:
`nvidia-smi` no funciona como en GPUs de escritorio. Es normal.

## 10. Errores comunes

* Instalar PyTorch estándar desde pip
* No activar `nvpmodel` y `jetson_clocks`
* Fuente de alimentación insuficiente
* Mezclar versiones de JetPack y librerías
* Esperar comportamiento tipo RTX

Esto es una SoC embebida.

## 11. Qué esperar de la GPU Orin Nano

* Muy buena para inferencia
* CUDA funcional
* TensorRT excelente
* Entrenamiento limitado
* Adecuada para visión, audio, LLM pequeños y edge AI

No es una GPU de datacenter.

## Próximos pasos recomendados

* PyTorch optimizado con TensorRT
* LLM pequeños combinando CPU y GPU
* Stack RAG acelerado específico para Orin Nano

@Autor: Marlon Cárdenas
2025
