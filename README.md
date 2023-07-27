# Trabajo de Fin de Grado - Manual

## Introducción

Este proyecto es parte del Trabajo de Fin de Grado. Se trata de un simulador que permite experimentar y evaluar diferente métricas de la 
reciente familia de autómatas celulares llamada Lenia, desarrollada por Bert Chan. También se incluye una de sus variantes, Flow Lenia, 
desarrollada por Plantec y colaboradores. Esta herramienta permite configurar los diferentes parámetros de estos autómatas celulares, 
así como evaluar diferentes métricas.

![image](https://github.com/G10on/LeniaPlus-Framework-Final-Project/assets/91230270/65c6adba-8ee6-459e-a1e9-a6e1706f1560)

## Paneles

Se diferencian 3 paneles principales: panel superior; visualizador; y panel de control. 

![image](https://github.com/G10on/LeniaPlus-Framework-Final-Project/assets/91230270/df759749-d478-4b82-a7c1-d5f3b6d7a4fb)

### Panel Superior

En el panel superior, se encuentran algunos de las operaciones que tiene todo simulador: play (reproducir), pasar al siguiente instante de tiempo, 
reiniciar la simulación, generar nuevos parámetros aleatorios a partir de una semilla, guardar el estado de la simulación, grabar un vídeo...

![image](https://github.com/G10on/LeniaPlus-Framework-Final-Project/assets/91230270/038aa992-0353-4909-987a-6a18b607d11e)

### Visualizador

En el visualizador, se puede observar la ejecución de la simulación en tiempo real. Se muestran hasta un máximo de 3 canales, debido a los canales RGB.

![image](https://github.com/G10on/LeniaPlus-Framework-Final-Project/assets/91230270/0588ff41-6510-4a8e-9ef7-90defbc7c3cc)

### Panel de Control

El panel de control se divide en tres ventanas: configuración de parámetros; galería de ejemplares; y estadísticas para análisis.

![image](https://github.com/G10on/LeniaPlus-Framework-Final-Project/assets/91230270/4cea1c6b-ebcf-4b41-b422-db1b8e2b4c46)

En la ventana de configuración de parámetros se puede establecer los valores de los parámetros del mundo, así como la estructura de cada kernel.
Cada kernel se puede configurar pulsando en el botón "Edit" al lado del kernel que se desa modificar. Esto abrirá una nueva ventana donde se puede 
modificar sus parámetros.

![image](https://github.com/G10on/LeniaPlus-Framework-Final-Project/assets/91230270/597c9d80-dd14-4b39-b434-8479e5637b47)

En la ventana de galería de ejemplares, se puede encontrar ejemplares añadidos por defecto, así como simulaciones que se han guardado manualmente.

![image](https://github.com/G10on/LeniaPlus-Framework-Final-Project/assets/91230270/b1104b9e-2c99-4b18-9ed8-f88fd0bb0522)

Por último, está la ventana de estadísticas para análisis, donde primero se muestra un pequeño panel que traza el movimiento de los individuos.
Debajo de este, se muestran las gráficas globales de las métricas de 'Supervivencia', 'Reproducción' y 'Morfología'.

![image](https://github.com/G10on/LeniaPlus-Framework-Final-Project/assets/91230270/485dbf6f-a769-4f56-8adf-3b224e227247)

Desde en esta misma ventana, se puede abrir otras dos ventanas emergentes que muestra el historial de registros de las métricas para cada individuo.

![image](https://github.com/G10on/LeniaPlus-Framework-Final-Project/assets/91230270/fa4bc145-175c-48ef-baea-d0b06c6d778a)



## Instalación

Se ha incluido un enlace al ejecutable de este proyecto, debido a que su tamaño excedía el límite de GitHub. Se ejecuta solo en sistemas operativos Linux, y se accede a través de un navegador Chrome. A continuación se indica cómo generar este ejecutable a partir del repositorio.

[Enlace para descargar el ejecutable](https://drive.google.com/file/d/1tdEt-k23cPFAlCe2pf1k9v7mPcyiWYzJ/view?usp=sharing)

Primero, se descarga o clona el repositorio en local. El entorno virtual de Python debe incluir las siguientes librerías: Numpy, Scipy, Jax, Eel, Pickle y PyInstaller.

Desde el directorio del repositorio, se abre una terminal, y se introduce el siguiente comando.

`python -m eel Processing/main.py web --onefile --noconsole`

Este comando generará una carpeta llamada 'dist', que contendrá el ejecutable 'main'. Se llama ejecuta este fichero (desde la terminal o con doble click). Por último, se accede desde el navegador Chrome a `localhost:8000`, que es el puerto por el que escucha el servidor de Eel. Para finalizar el servidor, se pulsa en el botón rojo en la parte superior derecha, y se cierra la ventana.