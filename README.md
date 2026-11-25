# Actividad 5 · Chatbot RAG documental Softtek

Isaac Esquivel Ruiz · Universidad Tecmilenio  
Sistemas y tecnologías de administración del conocimiento

## Objetivo y contexto
Consultar en lenguaje natural una base de 30 documentos simulados sobre soporte y operación de TI. Softtek es la referencia organizacional; los documentos no representan políticas internas reales. Se reutiliza el corpus y el recuperador de las actividades 3 y 4 y se agrega generación local.

## Enfoque
RAG con TF-IDF/coseno + **Qwen/Qwen3-0.6B**. No necesita una API key ni un servicio de inferencia de pago. La primera carga descarga el modelo público (aproximadamente 1.2 GB de pesos); se necesita internet y espacio disponible. Después utiliza la caché. CPU es válida, pero GPU acelera las respuestas. La carga en CPU usa float32, por lo que requiere más RAM que el tamaño descargado; se recomienda disponer de al menos 6 GB libres.

El corpus se conserva íntegro. Cada documento es corto y actúa como una unidad de recuperación; no se fragmenta. Se reutilizan TF-IDF con unigramas/bigramas y similitud coseno. Un ID explícito recupera el documento exacto; las recomendaciones recuperan Top-3 excluyendo la semilla. Para preguntas libres se recuperan hasta 3 documentos con score >= 0.07 (umbral heurístico, no probabilidad de certeza).

El LLM recibe la consulta y las fuentes seleccionadas y redacta la respuesta. **RAG no garantiza exactitud.** Las fuentes se muestran completas para revisarlas. El control de IDs rechaza referencias inexistentes, pero no verifica si cada afirmación está sustentada.


## Ejecutar la interfaz local
Python 3.10 o posterior. Desde la raíz de la carpeta:
```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```
Abre la dirección local indicada por Streamlit. La primera respuesta tarda más porque carga el modelo.

## Ejecutar en Colab
1. Sube `notebooks/Actividad_5_Chatbot_Softtek.ipynb`.
2. Ejecuta las celdas en orden. GPU es opcional.
3. El notebook incluye el corpus y los scripts, crea `actividad5_demo/`, y muestra las pruebas y sus resultados.
4. En la última sección escribe preguntas mediante el cuadro de texto. No requiere un túnel público de Streamlit para funcionar en Colab.

El notebook contiene copias de los archivos para ser autónomo; editar archivos externos no actualiza esas copias automáticamente. El notebook entregado tiene salidas de ejecución local; vuelve a ejecutarlo para registrar tu propia sesión en Colab.

## Pruebas y ejemplos
```bash
python pruebas.py
```
Preguntas en orden:
1. ¿De qué trata DOC008?
2. ¿Qué documentos relacionados existen?
3. ¿Cómo se realiza la restauración de un respaldo y cómo se verifica la integridad de los datos?
4. ¿Cuál es la órbita de Neptuno?
5. ¿De qué trata DOC999?

La pregunta 2 reutiliza el último ID explícito. La memoria guarda solo ese ID por sesión: no representa memoria conversacional general. “Nueva conversación” lo borra. El registro incluye respuestas literales, fuentes, tiempos, versiones y hash del corpus. Las verificaciones automáticas comprueban el funcionamiento; el reporte revisa la calidad semántica.

## Límites y mejoras
- TF-IDF depende del vocabulario; sinónimos o consultas ambiguas pueden recuperar fuentes equivocadas.
- Un modelo pequeño puede omitir o distorsionar detalles aun con documentos correctos.
- El filtro de similitud no detecta todos los casos sin respuesta, especialmente cuando comparten vocabulario con el corpus.
- El historial visual no se usa completo como contexto. Las referencias implícitas se resuelven con reglas limitadas.
- No hay permisos por rol, trazabilidad empresarial, verificador de afirmaciones ni garantía contra instrucciones maliciosas en documentos. Usar solo el corpus simulado para la actividad.
- Comparar modelos, recuperación híbrida y un conjunto más amplio de preguntas con evaluación humana antes de uso empresarial.

## Fuentes técnicas
- Modelo: https://huggingface.co/Qwen/Qwen3-0.6B
- Streamlit: https://docs.streamlit.io/develop/api-reference/chat/st.chat_input
- Corpus de referencia: https://github.com/ierisaac2005-png/Evaluacion_Softtek

No subas pesos descargados ni entornos virtuales a GitHub. El enlace del ejemplo de la plataforma del curso no fue accesible; la interfaz implementa un chat con entrada, historial y fuentes utilizando la documentación oficial.
