# Potatoinador

## Instalación

### Requisitos previos

Necesitas tener instalado [uv](https://docs.astral.sh/uv/). Es el gestor de paquetes y entorno que usamos en este proyecto.

### Pasos de instalación

1. **Crear el entorno virtual** (obligatorio):
   ```bash
   uv venv
   ```

2. **Activar el entorno virtual**:
   - En Linux:
     ```bash
     source .venv/bin/activate
     ```
   - En Windows:
     ```bash
     .venv\Scripts\activate
     ```

3. **Instalar las dependencias**:
   ```bash
   uv sync
   ```

**Importante:** El entorno virtual es obligatorio. No ejecutes el proyecto sin activarlo primero.

## Uso

El proyecto se ejecuta en tres pasos en este orden específico:

```bash
uv run prep_ds.py
uv run train.py
uv run main.py
```

### Detalles de cada paso

- **prep_ds.py**: Descarga y prepara los datasets necesarios (~1GB). Solo necesita ejecutarse la primera vez.
- **train.py**: Entrena el modelo. Genera archivos adicionales (~1GB aproximadamente).
- **main.py**: Ejecuta la aplicación principal con el modelo ya entrenado.

Cada script depende de los resultados del anterior, así que asegúrate de ejecutarlos en el orden indicado.
