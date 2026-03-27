"""
Скрипт для конвертации WFDB записи в EDF формат с аннотациями P, T зубцов и QRS комплексов.

Аннотации сохраняются как отдельные каналы событий в EDF файле.
"""

import os
import numpy as np
import wfdb
import pyedflib.highlevel as highlevel
from datetime import datetime


def create_ecg_with_annotations_edf(
    record_path: str,
    output_path: str,
    annotation_extension: str = 'atr'
):
    """
    Конвертирует WFDB запись в EDF файл с аннотациями P, T зубцов и QRS комплексов.
    
    Параметры:
        record_path: Путь к WFDB записи (без расширения)
        output_path: Путь для сохранения EDF файла
        annotation_extension: Расширение файла аннотаций (по умолчанию 'atr')
    """
    # Чтение записи WFDB
    print(f"Чтение записи: {record_path}")
    record = wfdb.rdrecord(record_path)
    
    # Чтение аннотаций
    print(f"Чтение аннотаций: {record_path}.{annotation_extension}")
    try:
        annotations = wfdb.rdann(record_path, extension=annotation_extension)
    except Exception as e:
        print(f"Предупреждение: Не удалось прочитать аннотации: {e}")
        annotations = None
    
    # Получение данных сигналов
    signals = record.p_signal if record.p_signal is not None else record.d_signal
    if signals.ndim == 1:
        signals = signals.reshape(-1, 1)
    
    fs = record.fs  # Частота дискретизации
    n_samples = signals.shape[0]
    n_channels = signals.shape[1]
    
    print(f"Частота дискретизации: {fs} Гц")
    print(f"Количество отсчетов: {n_samples}")
    print(f"Количество каналов: {n_channels}")
    
    # Имена сигналов
    signal_names = record.sig_name if record.sig_name else [f'CH{i+1}' for i in range(n_channels)]
    units = record.units if record.units else ['mV'] * n_channels
    
    # Подготовка физических минимумов и максимумов
    physical_min = [float(np.min(signals[:, i])) for i in range(n_channels)]
    physical_max = [float(np.max(signals[:, i])) for i in range(n_channels)]
    digital_min = [-32768] * n_channels
    digital_max = [32767] * n_channels
    
    # Извлечение аннотаций по типам
    p_samples = []
    qrs_samples = []
    t_samples = []
    
    if annotations:
        for i, symbol in enumerate(annotations.symbol):
            sample = int(annotations.sample[i])
            
            # Классификация аннотаций
            # '+' - P wave (зубец P)
            # 'N', 'A', 'V', 'L', 'R', 'B', 'j', 'E', 'J', 'S', 'F', 'f', 'Q', '=' - различные типы QRS
            # 'T' - T wave (зубец T)
            
            if symbol == '+':
                p_samples.append(sample)
            elif symbol in ['N', 'A', 'V', 'L', 'R', 'B', 'j', 'E', 'J', 'S', 'F', 'f', 'Q', '=']:
                qrs_samples.append(sample)
            elif symbol == 'T':
                t_samples.append(sample)
        
        print(f"\nНайдено аннотаций:")
        print(f"  P зубцов: {len(p_samples)}")
        print(f"  QRS комплексов: {len(qrs_samples)}")
        print(f"  T зубцов: {len(t_samples)}")
    
    # Создание списка сигналов для EDF
    signals_data = []
    signal_headers = []
    
    # Добавляем каналы ЭКГ
    for i in range(n_channels):
        signal_header = {
            'label': signal_names[i],
            'dimension': units[i],
            'sample_frequency': int(fs),
            'physical_min': physical_min[i],
            'physical_max': physical_max[i],
            'digital_min': digital_min[i],
            'digital_max': digital_max[i],
            'prefilter': '',
        }
        signal_headers.append(signal_header)
        signals_data.append(signals[:, i])
    
    # Добавляем каналы аннотаций как события
    # В EDF аннотации добавляются через special_headers
    
    # Подготовка заголовка EDF (используем правильные имена полей для pyedflib)
    # Требуемые поля: technician, recording_additional, patientname, patient_additional, 
    # patientcode, equipment, admincode, sex, startdate, birthdate
    edf_header = {
        'patientcode': '',
        'sex': '',
        'birthdate': datetime(1900, 1, 1),
        'patientname': '',
        'patient_additional': '',
        'startdate': datetime.now(),
        'technician': '',
        'equipment': '',
        'admincode': '',
        'recordingid': os.path.basename(record_path),
        'recording_additional': '',
    }
    
    # Создаем аннотации для EDF
    # Формат: список списков, где каждый внутренний список содержит кортежи (время_начала_сек, длительность_сек, описание)
    annotations_list = []
    
    # Аннотации P волн
    p_annotations = []
    for sample in p_samples:
        time_sec = sample / fs
        p_annotations.append((time_sec, 0, 'P'))
    
    # Аннотации QRS комплексов
    qrs_annotations = []
    for sample in qrs_samples:
        time_sec = sample / fs
        qrs_annotations.append((time_sec, 0, 'QRS'))
    
    # Аннотации T волн
    t_annotations = []
    for sample in t_samples:
        time_sec = sample / fs
        t_annotations.append((time_sec, 0, 'T'))
    
    # Сохраняем аннотации в специальный заголовок
    # Для pyedflib аннотации передаются через параметр annotations при записи
    
    # Запись EDF файла
    print(f"\nЗапись EDF файла: {output_path}")
    
    # Используем low-level API для добавления аннотаций
    import pyedflib
    
    # Открываем файл для записи
    f = pyedflib.EdfWriter(output_path, n_channels=n_channels, file_type=pyedflib.FILETYPE_EDFPLUS)
    
    # Устанавливаем заголовок сигнала
    channel_info = []
    for i in range(n_channels):
        ch_info = {
            'label': signal_names[i],
            'dimension': units[i],
            'sample_frequency': int(fs),  # Используем sample_frequency вместо sample_rate
            'physical_min': physical_min[i],
            'physical_max': physical_max[i],
            'digital_min': digital_min[i],
            'digital_max': digital_max[i],
            'prefilter': '',
        }
        channel_info.append(ch_info)
    
    f.setSignalHeaders(channel_info)
    
    # Устанавливаем заголовок записи
    f.setHeader(edf_header)
    
    # Записываем данные сигналов
    # Используем writeSamples для записи всех каналов сразу
    signals_list = [signals[:, i] for i in range(n_channels)]
    f.writeSamples(signals_list)
    
    # Добавляем аннотации
    # Аннотации должны быть отсортированы по времени
    all_annotations = []
    if p_annotations:
        all_annotations.extend(p_annotations)
    if qrs_annotations:
        all_annotations.extend(qrs_annotations)
    if t_annotations:
        all_annotations.extend(t_annotations)
    
    # Сортируем по времени
    all_annotations.sort(key=lambda x: x[0])
    
    # Записываем аннотации
    if all_annotations:
        for ann_time, duration, desc in all_annotations:
            f.writeAnnotation(ann_time, duration, desc)
    
    f.close()
    
    print(f"Конвертация завершена: {output_path}")
    return output_path


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 3:
        print("Использование:")
        print("  python annotate_ecg_to_edf.py <record_path> <output_edf_path>")
        print("\nПример:")
        print("  python annotate_ecg_to_edf.py /workspace/100 /workspace/100_annotated.edf")
        sys.exit(1)
    
    record_path = sys.argv[1]
    output_path = sys.argv[2]
    
    create_ecg_with_annotations_edf(record_path, output_path)
