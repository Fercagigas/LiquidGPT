import csv
import json
import argparse
import os
import torch
import random
import transformers
import time
import re
from tqdm import tqdm
import logging
import sys
import gc  # Para gestión de memoria
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configuración global
choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]
max_model_length = 4096
max_new_tokens = 2048
random.seed(12345)

def load_mmlu_pro():
    """
    Carga el dataset MMLU-Pro y lo preprocesa para su evaluación.

    Returns:
        tuple: Conjuntos de test y validación procesados
    """
    dataset = load_dataset("TIGER-Lab/MMLU-Pro")
    test_df, val_df = dataset["test"], dataset["validation"]
    test_df = preprocess(test_df)
    val_df = preprocess(val_df)
    return test_df, val_df


def load_model():
    """
    Carga el modelo y tokenizador usando Transformers en lugar de VLLM.
    Optimizado para gestionar recursos de GPU eficientemente en Windows.

    Returns:
        tuple: (model_tuple, tokenizer) donde model_tuple contiene (model, generation_config)
    """
    logging.info(f"Cargando modelo: {args.model}")

    # Configurar el tokenizador
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        use_fast=True  # Usa la versión más rápida del tokenizador si está disponible
    )

    # Asegurar que el tokenizador tenga configurados token de padding y EOS
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Configurar opciones de memoria según disponibilidad de GPU
    gpu_available = torch.cuda.is_available()
    device_map = "auto" if gpu_available else None
    memory_usage = float(args.gpu_util) if gpu_available else None

    logging.info(f"GPU disponible: {gpu_available}, Uso de memoria: {memory_usage}")

    # Cargar el modelo con configuraciones optimizadas
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if gpu_available else torch.float32,  # Usar precisión reducida en GPU
        device_map=device_map,
        max_memory={i: f"{int(float(args.gpu_util) * 100)}%" for i in range(torch.cuda.device_count())} if gpu_available else None,
        trust_remote_code=True,
        offload_folder="offload_folder",  # Carpeta para descargar tensores si es necesario
        offload_state_dict=True if gpu_available else False  # Permitir descarga si hay limitaciones de memoria
    )

    # Configuración de generación (equivalente a SamplingParams en VLLM)
    generation_config = transformers.GenerationConfig(
        temperature=0,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=False,
    )

    return (model, generation_config), tokenizer


def preprocess(test_df):
    """
    Preprocesa el dataset eliminando opciones no válidas.

    Args:
        test_df: Dataset a procesar

    Returns:
        list: Lista de ejemplos procesados
    """
    res_df = []
    for each in test_df:
        options = []
        for opt in each["options"]:
            if opt == "N/A":  # Omitir opciones no aplicables
                continue
            options.append(opt)
        each["options"] = options
        res_df.append(each)
    return res_df


def args_generate_path(input_args):
    """
    Genera la estructura de directorios para guardar resultados.

    Args:
        input_args: Argumentos de línea de comandos

    Returns:
        list: Componentes de la ruta de guardado
    """
    scoring_method = "CoT"  # Chain of Thought
    model_name = input_args.model.split("/")[-1]  # Extraer solo el nombre del modelo
    subjects = input_args.selected_subjects.replace(",", "-").replace(" ", "_")
    return [model_name, scoring_method, subjects]


def select_by_category(df, subject):
    """
    Filtra el dataset por una categoría específica.

    Args:
        df: Dataset a filtrar
        subject: Categoría por la que filtrar

    Returns:
        list: Dataset filtrado
    """
    res = []
    for each in df:
        if each["category"] == subject:
            res.append(each)
    return res


def format_cot_example(example, including_answer=True):
    """
    Formatea un ejemplo para prompting en formato Chain of Thought.

    Args:
        example: Ejemplo a formatear
        including_answer: Si se debe incluir la respuesta (para ejemplos de few-shot)

    Returns:
        str: Prompt formateado
    """
    prompt = "Question:\n"
    question = example["question"]
    options = example["options"]
    prompt += question + "\n"
    prompt += "Options:\n"
    for i, opt in enumerate(options):
        prompt += "{}. {}\n".format(choices[i], opt)
    if including_answer:
        # Formatea el contenido CoT para incluir "Answer: Let's think step by step"
        cot_content = example["cot_content"].replace("A: Let's think step by step.",
                                                     "Answer: Let's think step by step.")
        prompt += cot_content + "\n\n"
    else:
        prompt += "Answer: Let's think step by step."
    return prompt


def generate_cot_prompt(val_df, curr, k):
    """
    Genera un prompt completo con ejemplos few-shot para Chain of Thought.

    Args:
        val_df: Conjunto de validación
        curr: Ejemplo actual a evaluar
        k: Número de ejemplos few-shot a incluir

    Returns:
        str: Prompt completo con instrucciones y ejemplos
    """
    # Cargar plantilla de instrucciones
    prompt = ""
    with open(f"cot_prompt_lib/initial_prompt.txt", "r") as fi:
        for line in fi.readlines():
            prompt += line

    # Obtener categoría del ejemplo actual
    subject = curr["category"]

    # Filtrar ejemplos de validación por la misma categoría
    val_df = select_by_category(val_df, subject)
    val_df = val_df[: k]  # Tomar solo k ejemplos

    # Reemplazar marcador de categoría en la plantilla
    prompt = prompt.replace("{$}", subject) + "\n"

    # Añadir ejemplos few-shot con respuestas
    for example in val_df:
        prompt += format_cot_example(example, including_answer=True)

    # Añadir ejemplo actual sin respuesta
    prompt += format_cot_example(curr, including_answer=False)

    return prompt


def extract_answer(text):
    """
    Extrae la respuesta (letra A-J) del texto generado.
    Primer método: buscar patrones como "answer is (X)".

    Args:
        text: Texto generado por el modelo

    Returns:
        str o None: Letra de la respuesta extraída
    """
    pattern = r"answer is \(?([A-J])\)?"
    match = re.search(pattern, text.lower())
    if match:
        return match.group(1).upper()
    else:
        logging.debug("1st answer extract failed\n" + text)
        return extract_again(text)


def extract_again(text):
    """
    Segundo método de extracción: buscar patrones como "Answer: X".

    Args:
        text: Texto generado por el modelo

    Returns:
        str o None: Letra de la respuesta extraída
    """
    match = re.search(r'.*[aA]nswer:\s*([A-J])', text)
    if match:
        return match.group(1).upper()
    else:
        return extract_final(text)


def extract_final(text):
    """
    Método final de extracción: buscar la última ocurrencia de una letra A-J.

    Args:
        text: Texto generado por el modelo

    Returns:
        str o None: Letra de la respuesta extraída
    """
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return None


# Clase para detener la generación al encontrar un token específico
class StopOnTokens(transformers.StoppingCriteria):
    """
    Criterio de parada personalizado para detener la generación
    cuando se encuentra un token específico.

    Args:
        stop_token_ids: Lista de IDs de tokens donde detener la generación
    """
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids if isinstance(stop_token_ids, list) else [stop_token_ids]

    def __call__(self, input_ids, scores, **kwargs):
        for stop_id in self.stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


def batch_inference(model_tuple, generation_config, inference_batch, batch_size=1):
    """
    Realiza la inferencia en lotes para mejorar rendimiento.
    Versión optimizada para Transformers en lugar de VLLM.

    Args:
        model_tuple: Tupla (model, _) con el modelo cargado
        generation_config: Configuración de generación
        inference_batch: Lista de prompts para inferencia
        batch_size: Tamaño del lote (ajustar según memoria disponible)

    Returns:
        tuple: (pred_batch, response_batch) con predicciones y respuestas
    """
    model, _ = model_tuple
    model.eval()  # Establecer el modelo en modo evaluación
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Preparar la lista de criterios de parada para "Question:"
    stop_words = ["Question:", "question:"]
    stop_token_ids = []
    for word in stop_words:
        ids = tokenizer.encode(word, add_special_tokens=False)
        if ids:
            stop_token_ids.append(ids[0])

    response_batch = []
    pred_batch = []

    # Procesar en mini-lotes para optimizar memoria y rendimiento
    for i in tqdm(range(0, len(inference_batch), batch_size), desc="Procesando lotes"):
        current_batch = inference_batch[i:i+batch_size]

        try:
            # Tokenizar la entrada
            inputs = tokenizer(current_batch, return_tensors="pt", padding=True, truncation=True,
                             max_length=max_model_length - max_new_tokens)

            # Mover a GPU si está disponible
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # Configurar criterios de parada
            stopping_criteria = None
            if hasattr(transformers, 'StoppingCriteriaList'):
                stopping_criteria = transformers.StoppingCriteriaList([
                    StopOnTokens(stop_token_ids)
                ])

            # Generar respuestas
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0,
                    do_sample=False,
                    stopping_criteria=stopping_criteria
                )

            # Procesar cada salida del lote
            for j, output in enumerate(outputs):
                input_length = inputs["input_ids"].shape[1]

                # Decodificar solo la parte generada (no el prompt)
                generated_text = tokenizer.decode(output[input_length:], skip_special_tokens=True)
                response_batch.append(generated_text)

                # Extraer la respuesta
                pred = extract_answer(generated_text)
                pred_batch.append(pred)

                # Registrar la respuesta para depuración
                if pred is None:
                    logging.warning(f"No se pudo extraer respuesta para el ejemplo {i+j}")
                    logging.debug(f"Texto generado: {generated_text[:100]}...")

            # Liberar memoria después de cada lote
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            logging.error(f"Error en lote {i}: {str(e)}")
            # Añadir respuestas nulas en caso de error para mantener el orden
            for _ in range(len(current_batch)):
                response_batch.append("")
                pred_batch.append(None)

    logging.info(f"Batch de {len(inference_batch)} ejemplos procesado en {time.time() - start:.2f} segundos")
    return pred_batch, response_batch


def save_res(res, output_path):
    """
    Guarda los resultados y calcula las métricas de precisión.

    Args:
        res: Lista de resultados con predicciones
        output_path: Ruta para guardar los resultados

    Returns:
        tuple: (accuracy, correct_count, wrong_count)
    """
    accu, corr, wrong = 0.0, 0.0, 0.0

    # Guardar resultados en formato JSON
    with open(output_path, "w") as fo:
        fo.write(json.dumps(res, indent=2))

    # Evaluar precisión
    for each in res:
        # Manejar caso donde no se extrajo predicción
        if not each["pred"]:
            x = random.randint(0, len(each["options"]) - 1)
            if x == each["answer_index"]:
                corr += 1
            else:
                wrong += 1
        # Caso donde la predicción coincide con la respuesta
        elif each["pred"] == each["answer"]:
            corr += 1
        # Caso donde la predicción es incorrecta
        else:
            wrong += 1

    # Calcular precisión
    if corr + wrong == 0:
        return 0.0, 0.0, 0.0
    accu = corr / (corr + wrong)

    return accu, corr, wrong


@torch.no_grad()  # Desactivar cálculo de gradientes para ahorrar memoria
def eval_cot(subject, model, tokenizer, val_df, test_df, output_path):
    """
    Evalúa el rendimiento del modelo en una categoría específica usando Chain of Thought.

    Args:
        subject: Categoría a evaluar
        model: Modelo cargado
        tokenizer: Tokenizador
        val_df: Conjunto de validación
        test_df: Conjunto de prueba
        output_path: Ruta para guardar resultados

    Returns:
        tuple: (accuracy, correct_count, wrong_count)
    """
    llm, sampling_params = model
    global choices
    logging.info(f"Evaluando categoría: {subject}")
    inference_batches = []

    # Preparar los prompts para cada ejemplo de prueba
    for i in tqdm(range(len(test_df)), desc=f"Preparando prompts para {subject}"):
        k = args.ntrain  # Número de ejemplos few-shot
        curr = test_df[i]
        prompt_length_ok = False
        prompt = None

        # Reducir k hasta que el prompt quepa en el contexto del modelo
        while not prompt_length_ok:
            prompt = generate_cot_prompt(val_df, curr, k)
            inputs = tokenizer(prompt, return_tensors="pt")

            # Mover a GPU si está disponible
            if torch.cuda.is_available():
                inputs = {key: value.cuda() for key, value in inputs.items()}

            # Verificar longitud
            length = len(inputs["input_ids"][0])
            if length < max_model_length - max_new_tokens:
                prompt_length_ok = True
            else:
                k -= 1
                if k < 0:
                    logging.warning(f"No se pudo reducir suficiente el prompt para ejemplo {i}")
                    # Truncar como último recurso
                    prompt = tokenizer.decode(inputs["input_ids"][0][:max_model_length - max_new_tokens - 50])
                    prompt_length_ok = True

        inference_batches.append(prompt)

    # Determinar tamaño de lote según disponibilidad de memoria
    batch_size = 1  # Valor por defecto conservador
    if torch.cuda.is_available():
        # Ajustar según memoria disponible (más conservador para modelos grandes)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        if gpu_memory > 24:
            batch_size = 4
        elif gpu_memory > 16:
            batch_size = 2

    logging.info(f"Usando tamaño de lote: {batch_size}")

    # Realizar inferencia
    pred_batch, response_batch = batch_inference(model, sampling_params, inference_batches, batch_size)

    # Preparar resultados
    res = []
    for j, curr in enumerate(test_df):
        curr["pred"] = pred_batch[j]
        curr["model_outputs"] = response_batch[j]
        res.append(curr)

    # Calcular y guardar métricas
    accu, corr, wrong = save_res(res, output_path)
    logging.info(f"Precisión para {subject}: {accu:.4f}, correctas: {corr}, incorrectas: {wrong}")

    return accu, corr, wrong


def main():
    """
    Función principal que coordina todo el proceso de evaluación.
    Carga el modelo, datasets, ejecuta evaluaciones y guarda resultados.
    """
    # Cargar modelo y tokenizador
    model, tokenizer = load_model()

    # Crear directorio de resultados si no existe
    if not os.path.exists(save_result_dir):
        os.makedirs(save_result_dir)

    # Cargar conjuntos de datos
    logging.info("Cargando datasets MMLU-Pro...")
    full_test_df, full_val_df = load_mmlu_pro()

    # Determinar las categorías a evaluar
    all_subjects = []
    for each in full_test_df:
        if each["category"] not in all_subjects:
            all_subjects.append(each["category"])

    # Filtrar categorías según parámetros
    if args.selected_subjects == "all":
        selected_subjects = all_subjects
    else:
        selected_subjects = []
        args_selected = args.selected_subjects.split(",")
        for sub in all_subjects:
            for each in args_selected:
                if each.replace(" ", "_") in sub.replace(" ", "_"):
                    selected_subjects.append(sub)

    # Mostrar categorías seleccionadas
    logging.info("Categorías seleccionadas:\n" + "\n".join(selected_subjects))
    print("Categorías seleccionadas:\n" + "\n".join(selected_subjects))

    # Preparar diccionario para almacenar estadísticas
    sta_dict = {}
    selected_subjects = sorted(selected_subjects)

    # Iniciar archivo de resumen
    with open(os.path.join(summary_path), 'a') as f:
        f.write("\n------category level sta------\n")

    # Evaluar cada categoría
    for subject in selected_subjects:
        if subject not in sta_dict:
            sta_dict[subject] = {"corr": 0.0, "wrong": 0.0, "accu": 0.0}

        # Filtrar datasets por categoría
        test_df = select_by_category(full_test_df, subject)
        val_df = select_by_category(full_val_df, subject)

        # Preparar ruta de salida
        output_path = os.path.join(save_result_dir, "{}.json".format(subject))

        # Evaluar categoría
        acc, corr_count, wrong_count = eval_cot(subject, model, tokenizer, val_df, test_df, output_path)

        # Guardar estadísticas
        sta_dict[subject]["corr"] = corr_count
        sta_dict[subject]["wrong"] = wrong_count
        sta_dict[subject]["accu"] = acc

        # Escribir en archivo de resumen
        with open(os.path.join(summary_path), 'a') as f:
            f.write("Average accuracy {:.4f} - {}\n".format(sta_dict[subject]["accu"], subject))

    # Calcular estadísticas globales
    total_corr, total_wrong = 0.0, 0.0
    for k, v in sta_dict.items():
        total_corr += v["corr"]
        total_wrong += v["wrong"]
    total_accu = total_corr / (total_corr + total_wrong + 0.000001)
    sta_dict["total"] = {"corr": total_corr, "wrong": total_wrong, "accu": total_accu}

    # Escribir resumen final
    with open(os.path.join(summary_path), 'a') as f:
        f.write("\n------average acc sta------\n")
        weighted_acc = total_accu
        f.write("Average accuracy: {:.4f}\n".format(weighted_acc))

    # Registrar en archivo de seguimiento global
    with open(global_record_file, 'a', newline='') as file:
        writer = csv.writer(file)
        record = args_generate_path(args) + [time_str, weighted_acc]
        writer.writerow(record)

    logging.info(f"Evaluación completada. Precisión global: {total_accu:.4f}")


if __name__ == "__main__":
    # Configuración de argumentos de línea de comandos
    parser = argparse.ArgumentParser(description="Evaluación de modelos de lenguaje en MMLU-Pro usando Chain of Thought")
    parser.add_argument("--ntrain", "-k", type=int, default=5,
                      help="Número de ejemplos few-shot a incluir en el prompt")
    parser.add_argument("--selected_subjects", "-sub", type=str, default="all",
                      help="Categorías a evaluar, separadas por comas, o 'all' para todas")
    parser.add_argument("--save_dir", "-s", type=str, default="results",
                      help="Directorio para guardar resultados")
    parser.add_argument("--global_record_file", "-grf", type=str,
                      default="eval_record_collection.csv",
                      help="Archivo CSV para registro global de evaluaciones")
    parser.add_argument("--gpu_util", "-gu", type=str, default="0.8",
                      help="Fracción de memoria GPU a utilizar (0.0-1.0)")
    parser.add_argument("--model", "-m", type=str, default="meta-llama/Llama-2-7b-hf",
                      help="Ruta o nombre del modelo a evaluar")

    args = parser.parse_args()

    # Crear directorios necesarios
    os.makedirs(args.save_dir, exist_ok=True)
    global_record_file = args.global_record_file

    # Configurar rutas para guardar resultados
    save_result_dir = os.path.join(
        args.save_dir, "/".join(args_generate_path(args))
    )
    file_prefix = "-".join(args_generate_path(args))
    timestamp = time.time()
    time_str = time.strftime('%m-%d_%H-%M', time.localtime(timestamp))
    file_name = f"{file_prefix}_{time_str}_summary.txt"
    summary_path = os.path.join(args.save_dir, "summary", file_name)

    # Crear directorios adicionales
    os.makedirs(os.path.join(args.save_dir, "summary"), exist_ok=True)
    os.makedirs(save_result_dir, exist_ok=True)
    save_log_dir = os.path.join(args.save_dir, "log")
    os.makedirs(save_log_dir, exist_ok=True)

    # Configurar logging
    log_filename = os.path.join(save_log_dir, file_name.replace("_summary.txt", "_logfile.log"))
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Registrar inicio de ejecución
    logging.info(f"Iniciando evaluación con modelo: {args.model}")

    # Ejecutar evaluación
    main()