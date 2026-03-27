"""
Модуль для конвертации файлов из формата WFDB в формат EDF.

WFDB (WaveForm DataBase) - формат для хранения физиологических сигналов (например, ЭКГ).
EDF (European Data Format) - универсальный формат для обмена медицинскими данными.

Зависимости:
    - wfdb: для чтения WFDB файлов
    - pyedflib: для записи EDF файлов
    - numpy: для обработки данных

Пример использования:
    # Конвертировать все файлы из папки
    from wfdb_to_edf import convert_folder
    
    convert_folder('/path/to/wfdb', '/output/edf')
    
    # Или через командную строку:
    # python wfdb_to_edf.py /path/to/wfdb /output/edf
"""

import os
import glob
from datetime import datetime
from typing import Optional, Union

import numpy as np
import wfdb
import pyedflib.highlevel as highlevel


def convert_single_file(
    record_path: str,
    output_path: Optional[str] = None,
    signal_names: Optional[list[str]] = None,
    physical_min: Optional[list[float]] = None,
    physical_max: Optional[list[float]] = None,
    digital_min: Optional[list[int]] = None,
    digital_max: Optional[list[int]] = None,
    patient_info: Optional[dict] = None,
    recording_info: Optional[dict] = None,
) -> str:
    """
    Конвертирует один файл из формата WFDB в EDF.

    Параметры:
        record_path: Путь к файлу записи WFDB (без расширения или с расширением .dat/.hea)
        output_path: Путь для сохранения EDF файла. Если None, используется имя исходного файла с расширением .edf
        signal_names: Список имен сигналов для EDF заголовка. Если None, используются имена из WFDB
        physical_min: Минимальные физические значения для каждого сигнала (в единицах измерения, например мВ)
        physical_max: Максимальные физические значения для каждого сигнала
        digital_min: Минимальные цифровые значения (обычно -32768 для 16-бит)
        digital_max: Максимальные цифровые значения (обычно 32767 для 16-бит)
        patient_info: Словарь с информацией о пациенте:
            - patientcode: код пациента
            - sex: пол ('M' или 'F')
            - birthdate: дата рождения (datetime.date или строка 'DD-MMM-YYYY')
            - additional: дополнительная информация
        recording_info: Словарь с информацией о записи:
            - startdate: дата начала записи (datetime.date или строка 'DD-MMM-YYYY')
            - starttime: время начала записи (datetime.time или строка 'HH:MM:SS')
            - technician: техник
            - equipment: оборудование
            - admincode: административный код
            - recordingid: ID записи
            - additional: дополнительная информация

    Возвращает:
        Путь к созданному EDF файлу
    """
    # Чтение записи WFDB
    if record_path.endswith('.hea'):
        record_path = record_path[:-4]
    elif record_path.endswith('.dat'):
        record_path = record_path[:-4]

    record = wfdb.rdrecord(record_path)
    
    # Получение аннотаций (если есть)
    try:
        annotations = wfdb.rdann(record_path, extension='atr')
    except Exception:
        annotations = None

    # Подготовка данных сигналов
    signals = record.p_signal if record.p_signal is not None else record.d_signal
    if signals.ndim == 1:
        signals = signals.reshape(-1, 1)
    
    fs = record.fs  # Частота дискретизации
    
    # Имена сигналов
    if signal_names is None:
        signal_names = record.sig_name if record.sig_name else [f'CH{i+1}' for i in range(signals.shape[1])]
    
    # Единицы измерения
    units = record.units if record.units else ['mV'] * signals.shape[1]
    
    # Физические минимумы и максимумы
    n_signals = signals.shape[1]
    
    if physical_min is None:
        physical_min = [float(np.min(signals[:, i])) for i in range(n_signals)]
    if physical_max is None:
        physical_max = [float(np.max(signals[:, i])) for i in range(n_signals)]
    if digital_min is None:
        digital_min = [-32768] * n_signals
    if digital_max is None:
        digital_max = [32767] * n_signals
    
    # Обеспечиваем одинаковую длину списков
    while len(signal_names) < n_signals:
        signal_names.append(f'CH{len(signal_names)+1}')
    while len(units) < n_signals:
        units.append('mV')
    while len(physical_min) < n_signals:
        physical_min.append(float(np.min(signals[:, len(physical_min)])))
    while len(physical_max) < n_signals:
        physical_max.append(float(np.max(signals[:, len(physical_max)])))
    while len(digital_min) < n_signals:
        digital_min.append(-32768)
    while len(digital_max) < n_signals:
        digital_max.append(32767)
    
    # Подготовка информации о пациенте
    if patient_info is None:
        patient_info = {}
    
    patientcode = patient_info.get('patientcode', '')
    sex = patient_info.get('sex', '')
    birthdate = patient_info.get('birthdate', '')
    patient_additional = patient_info.get('additional', '')
    
    # Преобразование даты рождения в формат EDF
    if isinstance(birthdate, datetime):
        birthdate = birthdate.strftime('%d-%b-%Y').upper()
    elif isinstance(birthdate, str) and birthdate:
        pass  # Оставляем как есть
    else:
        birthdate = ''
    
    # Подготовка информации о записи
    if recording_info is None:
        recording_info = {}
    
    startdate = recording_info.get('startdate', '')
    starttime = recording_info.get('starttime', '')
    technician = recording_info.get('technician', '')
    equipment = recording_info.get('equipment', '')
    admincode = recording_info.get('admincode', '')
    recordingid = recording_info.get('recordingid', '')
    recording_additional = recording_info.get('additional', '')
    
    # Преобразование даты и времени начала в формат EDF
    if isinstance(startdate, datetime):
        startdate = startdate.strftime('%d-%b-%Y').upper()
    elif isinstance(startdate, str) and startdate:
        pass
    else:
        startdate = ''
    
    if isinstance(starttime, datetime):
        starttime = starttime.strftime('%H:%M:%S')
    elif isinstance(starttime, str) and starttime:
        pass
    else:
        starttime = ''
    
    # Определение пути выхода
    if output_path is None:
        base_name = os.path.basename(record_path)
        output_dir = os.path.dirname(record_path)
        output_path = os.path.join(output_dir, f'{base_name}.edf')
    elif not output_path.endswith('.edf'):
        output_path = output_path + '.edf'
    
    # Создание списка сигналов для EDF
    signals_data = []
    signal_headers = []
    for i in range(n_signals):
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
    
    # Подготовка заголовка EDF
    edf_header = {
        'patientcode': patientcode,
        'sex': sex,
        'birthdate': birthdate if isinstance(birthdate, datetime) else datetime(1900, 1, 1),
        'patient_additional': patient_additional,
        'startdate': startdate if isinstance(startdate, datetime) else datetime.now(),
        'starttime': starttime,
        'technician': technician,
        'equipment': equipment,
        'admincode': admincode,
        'recordingid': recordingid,
        'recording_additional': recording_additional,
    }
    
    # Запись EDF файла
    highlevel.write_edf(
        output_path,
        signals=signals_data,
        signal_headers=signal_headers,
        header=edf_header,
    )
    
    print(f"Конвертация завершена: {output_path}")
    return output_path


def convert_folder(
    input_folder: str,
    output_folder: str,
    pattern: str = '*.hea',
    recursive: bool = False,
    **kwargs
) -> list[str]:
    """
    Конвертирует все WFDB файлы из папки в формат EDF.

    Параметры:
        input_folder: Путь к папке с исходными WFDB файлами
        output_folder: Путь к папке для сохранения EDF файлов
        pattern: Шаблон для поиска файлов (по умолчанию '*.hea')
        recursive: Если True, искать файлы рекурсивно во всех подпапках
        **kwargs: Дополнительные параметры, передаваемые в convert_single_file

    Возвращает:
        Список путей к созданным EDF файлам
    """
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Папка не найдена: {input_folder}")
    
    # Создаем выходную папку если она не существует
    os.makedirs(output_folder, exist_ok=True)
    
    # Поиск файлов
    if recursive:
        pattern_full = os.path.join(input_folder, '**', pattern)
        files = glob.glob(pattern_full, recursive=True)
    else:
        pattern_full = os.path.join(input_folder, pattern)
        files = glob.glob(pattern_full)
    
    if not files:
        print(f"Файлы не найдены по шаблону: {pattern_full}")
        return []
    
    converted_files = []
    
    for file_path in files:
        # Получаем базовое имя файла (без расширения)
        base_name = os.path.basename(file_path)
        if base_name.endswith('.hea'):
            base_name = base_name[:-4]
        elif base_name.endswith('.dat'):
            base_name = base_name[:-4]
        
        # Формируем путь выхода
        output_path = os.path.join(output_folder, f'{base_name}.edf')
        
        try:
            convert_single_file(
                record_path=file_path,
                output_path=output_path,
                **kwargs
            )
            converted_files.append(output_path)
        except Exception as e:
            print(f"Ошибка при конвертации файла {file_path}: {e}")
    
    print(f"\nКонвертация завершена. Создано файлов: {len(converted_files)}")
    return converted_files


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 3:
        print("Использование:")
        print("  python wfdb_to_edf.py <input_folder> <output_folder> [--recursive]")
        print("\nПример:")
        print("  python wfdb_to_edf.py /path/to/wfdb /output/edf")
        print("  python wfdb_to_edf.py /path/to/wfdb /output/edf --recursive")
        sys.exit(1)
    
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    recursive = '--recursive' in sys.argv
    
    convert_folder(input_folder, output_folder, recursive=recursive)
