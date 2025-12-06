# 📊 TFM Grupo 06 - Proyecto de Data Science

Repositorio oficial para el Trabajo Fin de Máster (TFM). Este proyecto integra ingeniería de datos, análisis exploratorio, modelado predictivo y visualización en Power BI.

---

## 📂 Estructura del Proyecto

La organización de carpetas sigue un estándar híbrido adaptado a nuestras entregas y flujo MLOps:

```text
TFM_06/
├── 📁 Actas_Convocatorias/        # Actas y notas de reuniones.
├── 📁 DatasetsTFM/                # Datos RAW y procesados (Gestionado por DVC).
│   └── diccionario/               # Diccionarios de variables y metadatos.
├── 📁 Documentación_General/      # Documentación de alcance.
├── 📁 Documentación_Técnica/      # Guías técnicas (DVC, Jupytext, etc).
├── 📁 Entregables.../             # Carpetas para entregas oficiales del máster.
├── 📁 src/                        # Código fuente modular (.py).
│   └── utils.py                   # Funciones comunes reutilizables.
├── .dvc/                          # Configuración interna de Data Version Control.
├── jupytext.toml                  # Configuración de sincronización Notebook-Script.
├── requirements.txt               # Dependencias del proyecto.
└── README.md                      # Este archivo.
```

---

## 🚀 Guía de Inicio Rápido
Sigue estos pasos para configurar tu entorno local.

1. Clonar y Entorno Virtual
```bash

git clone [https://github.com/Enrique737/TFM_06.git](https://github.com/Enrique737/TFM_06.git)
cd TFM_06

# Crear entorno virtual (si no usas conda)
python -m venv penv
# Activar entorno (Windows)
.\penv\Scripts\activate
```

2. Instalar Dependencias
```Bash

pip install -r requirements.txt
```

3. Descargar los Datos (DVC)
Los datasets pesados (>100MB) no están en GitHub. Están alojados en Google Drive y gestionados por DVC. Para bajarlos:

```Bash

# Esto leerá los archivos .dvc y descargará los .csv/.parquet reales
dvc pull
```
Nota: La primera vez te pedirá autenticación de Google en el navegador.

---
## 🛠️ Flujo de Trabajo (Workflow)
Para evitar conflictos y mantener el repositorio limpio, seguimos estas reglas estrictas.

1. Jupyter Notebooks y Jupytext
**Problema**: Los .ipynb generan conflictos horribles en Git. Solución: Usamos Jupytext.

- **Git ignora los .ipynb**: Solo subimos los archivos .py pareados.

- **Al trabajar**:

  - Abres el archivo .py en Jupyter (click derecho -> Open with Notebook).
  - Trabajas normal y guardas. Jupytext actualiza el .py automáticamente.
  - Haces git add archivo.py (el notebook se ignora).

- **Al recibir cambios (**`git pull`**)**:

  - Recibes un .py nuevo.
  - Al abrirlo en Jupyter, el notebook se regenera con el código nuevo (pero sin outputs).

2. Gestión de Datos (DVC)
Nunca subas archivos CSV, XLSX o PKL grandes directamente con git add.

**Para bajar datos actualizados:**

```Bash

dvc pull
```
**Para subir un nuevo dataset:**

```Bash

# 1. DVC trackea el archivo pesado
dvc add DatasetsTFM/nuevo_archivo.csv

# 2. Subimos los datos a la nube
dvc push

# 3. Git trackea solo el puntero (.dvc)
git add DatasetsTFM/nuevo_archivo.csv.dvc .gitignore
git commit -m "Add: nuevo dataset"
git push
```

---
## 🌳 Estrategia de Ramas (Git Flow)
- main: 🛡️ Código estable y producción. Nadie hace commit directo aquí.
- feat/nombre-tarea: ✨ Para nuevos análisis, modelos o limpieza.
  - Ej: feat/limpieza-nulos, feat/modelo-churn
- fix/nombre-bug: 🐛 Para corregir errores.

**Regla de Oro:** Se trabaja en ramas y se hace Pull Request (PR) a main.

---

## 🤝 Contribución
1. Actualiza tu rama: git pull origin main
2. Crea tu rama: git checkout -b feat/mi-analisis
3. Si instalas librerías nuevas, actualiza el requirements: pipreqs . --force
4. Sube cambios y abre PR.