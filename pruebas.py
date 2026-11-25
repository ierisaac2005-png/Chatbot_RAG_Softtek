"""Ejecuta tres consultas requeridas y controles adicionales con el modelo real."""
from pathlib import Path
import sys,json,time,hashlib,platform
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from chatbot import ChatbotRAG, MODEL_ID, MODEL_REVISION
import pandas as pd
import transformers,torch,sklearn
CASOS=[
 ('P1','Información general','¿De qué trata DOC008?'),
 ('P2','Recomendación y seguimiento','¿Qué documentos relacionados existen?'),
 ('P3','Consulta libre','¿Cómo se realiza la restauración de un respaldo y cómo se verifica la integridad de los datos?'),
 ('P4','Sin cobertura','¿Cuál es la órbita de Neptuno?'),
 ('P5','ID inválido','¿De qué trata DOC999?'),
]
def ejecutar():
    bot=ChatbotRAG();inicio=time.perf_counter();bot.cargar_modelo()
    carga=round(time.perf_counter()-inicio,3)
    registros=[];activo=None
    for pid,tipo,pregunta in CASOS:
        r=bot.responder(pregunta,activo);activo=r.documento_activo
        registro={'prueba':pid,'tipo':tipo,**r.to_dict()};registros.append(registro)
        print(f"\n{pid} | {tipo}\nENTRADA: {pregunta}\nMODO: {r.modo}\nRESPUESTA: {r.respuesta}\nFUENTES: {[f['id'] for f in r.fuentes]}\nTIEMPO: {r.segundos}s",flush=True)
    # Verificaciones estructurales; no califican la veracidad de la generación.
    assert registros[0]['fuentes'][0]['id']=='DOC008'
    assert 'DOC008' not in [f['id'] for f in registros[1]['fuentes']]
    assert registros[3]['modo']=='sin_contexto'
    assert registros[4]['modo']=='id_invalido'
    assert all(r['modo']=='rag' for r in registros[:3])
    destino=ROOT/'evidencias';destino.mkdir(exist_ok=True)
    resultado={'modelo':MODEL_ID,'revision':MODEL_REVISION,'segundos_carga':carga,
               'corpus_sha256':hashlib.sha256((ROOT/'data/corpus.csv').read_bytes()).hexdigest(),
               'versiones':{'python':platform.python_version(),'transformers':transformers.__version__,'torch':torch.__version__,'sklearn':sklearn.__version__},
               'pruebas':registros,'verificaciones':'OK; revisión semántica manual en el reporte'}
    (destino/'resultados_pruebas.json').write_text(json.dumps(resultado,ensure_ascii=False,indent=2),encoding='utf-8')
    pd.DataFrame([{'prueba':r['prueba'],'tipo':r['tipo'],'pregunta':r['pregunta'],'respuesta':r['respuesta'],
                  'fuentes':', '.join(f['id'] for f in r['fuentes']),'modo':r['modo'],'segundos':r['segundos']} for r in registros]).to_csv(destino/'tabla_pruebas.csv',index=False)
    print('\nPRUEBAS COMPLETADAS. Verificaciones estructurales: OK.',flush=True)
    return resultado
if __name__=='__main__':ejecutar()
