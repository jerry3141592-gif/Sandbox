"""
Модуль для конвертации WFDB аннотаций в формат CSV.

WFDB (WaveForm DataBase) - библиотека для работы с физиологическими сигналами,
особенно с данными из PhysioNet (например, база данных MIT-BIH Arrhythmia).
"""

import os
import csv
from pathlib import Path
from typing import Optional, Union
import wfdb


def convert_annotation_to_csv(annotation: wfdb.Annotation, output_path: str) -> None:
    """
    Конвертирует один объект аннотации WFDB в CSV файл.

    Параметры:
        annotation: Объект wfdb.Annotation, содержащий разметку.
        output_path: Путь к выходному CSV файлу.

    Формат выходного CSV:
        - sample: номер сэмпла (отсчет)
        - symbol: символ аннотации (основная метка типа события)
        - subtype: подтип аннотации (если доступен)
        - chan: канал (если доступен)
        - num: номер аннотации (если доступен)
        - aux_note: дополнительная заметка (если доступна)
    """
    # Определяем количество аннотаций
    n_annotations = len(annotation.sample) if annotation.sample is not None else 0
    
    # Подготавливаем данные для записи
    data = []
    for i in range(n_annotations):
        row = {
            'sample': int(annotation.sample[i]) if annotation.sample is not None else None,
            'symbol': annotation.symbol[i] if annotation.symbol else None,
            'subtype': int(annotation.subtype[i]) if annotation.subtype is not None and len(annotation.subtype) > i else None,
            'chan': int(annotation.chan[i]) if annotation.chan is not None and len(annotation.chan) > i else None,
            'num': int(annotation.num[i]) if annotation.num is not None and len(annotation.num) > i else None,
            'aux_note': annotation.aux_note[i] if annotation.aux_note and len(annotation.aux_note) > i else None,
        }
        data.append(row)
    
    # Записываем в CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['sample', 'symbol', 'subtype', 'chan', 'num', 'aux_note']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(data)
    
    print(f"Аннотация сохранена в: {output_path}")


def convert_folder_annotations(
    input_folder: Union[str, Path],
    output_folder: Optional[Union[str, Path]] = None,
    extension: str = '.atr'
) -> None:
    """
    Конвертирует все WFDB аннотации из папки в формат CSV.

    Параметры:
        input_folder: Путь к папке с файлами WFDB (.dat, .hea, .atr и т.д.).
        output_folder: Путь к папке для сохранения CSV файлов.
                       Если не указан, создается подпапка 'csv_annotations' 
                       внутри input_folder.
        extension: Расширение файлов аннотаций для поиска (по умолчанию '.atr').
                   Может быть также '.qrsc', '.event' и др.

    Пример использования:
        # Конвертировать все аннотации из папки
        convert_folder_annotations('/path/to/wfdb/files')
        
        # С указанием выходной папки
        convert_folder_annotations('/path/to/input', '/path/to/output')
        
        # С другим расширением аннотаций
        convert_folder_annotations('/path/to/input', extension='.qrsc')
    """
    import os
    
    input_path = Path(input_folder).resolve()
    
    if not input_path.exists():
        raise FileNotFoundError(f"Папка не найдена: {input_path}")
    
    if not input_path.is_dir():
        raise NotADirectoryError(f"Указанный путь не является папкой: {input_path}")
    
    # Определяем выходную папку
    if output_folder is None:
        output_path = input_path / 'csv_annotations'
    else:
        output_path = Path(output_folder).resolve()
    
    # Создаем выходную папку если она не существует
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Находим все файлы записей (.hea файлы)
    hea_files = list(input_path.glob('*.hea'))
    
    if not hea_files:
        print(f"Файлы .hea не найдены в папке: {input_path}")
        return
    
    print(f"Найдено файлов записей: {len(hea_files)}")
    
    converted_count = 0
    error_count = 0
    
    # Сохраняем текущую рабочую директорию
    original_dir = os.getcwd()
    
    try:
        # Переходим в папку с данными для корректной работы wfdb.rdann
        os.chdir(str(input_path))
        
        for hea_file in hea_files:
            try:
                # Получаем имя записи без расширения
                record_name = hea_file.stem
                
                # Читаем аннотацию (работаем из папки с данными)
                annotation = wfdb.rdann(
                    record_name=str(record_name),
                    extension=extension.lstrip('.')
                )
                
                # Формируем имя выходного файла
                csv_filename = f"{record_name}_{extension.lstrip('.')}.csv"
                csv_filepath = output_path / csv_filename
                
                # Конвертируем в CSV
                convert_annotation_to_csv(annotation, str(csv_filepath))
                converted_count += 1
                
            except FileNotFoundError:
                print(f"Предупреждение: Файл аннотации не найден для записи {hea_file.stem}")
                error_count += 1
            except Exception as e:
                print(f"Ошибка при обработке файла {hea_file.name}: {e}")
                error_count += 1
    finally:
        # Возвращаемся в исходную директорию
        os.chdir(original_dir)
    
    print(f"\n=== Результат ===")
    print(f"Всего обработано записей: {len(hea_files)}")
    print(f"Успешно сконвертировано: {converted_count}")
    print(f"Ошибок: {error_count}")
    print(f"CSV файлы сохранены в: {output_path}")


def convert_single_file(
    record_path: Union[str, Path],
    output_csv_path: Optional[Union[str, Path]] = None,
    extension: str = '.atr'
) -> str:
    """
    Конвертирует одну WFDB запись в CSV.

    Параметры:
        record_path: Путь к файлу записи (.hea) или имя записи без расширения.
        output_csv_path: Путь к выходному CSV файлу.
                        Если не указан, создается рядом с исходным файлом.
        extension: Расширение файла аннотаций (по умолчанию '.atr').

    Возвращает:
        Путь к созданному CSV файлу.
    """
    import os
    
    record_path = Path(record_path).resolve()
    
    # Определяем путь к записи и имя записи
    if record_path.suffix == '.hea':
        dir_name = record_path.parent
        record_name = record_path.stem
    else:
        dir_name = record_path.parent if record_path.parent else Path('.')
        record_name = record_path.name
    
    # Сохраняем текущую рабочую директорию
    original_dir = os.getcwd()
    
    try:
        # Переходим в папку с данными
        os.chdir(str(dir_name))
        
        # Читаем аннотацию
        annotation = wfdb.rdann(
            record_name=str(record_name),
            extension=extension.lstrip('.')
        )
        
        # Определяем путь для CSV
        if output_csv_path is None:
            output_csv_path = dir_name / f"{record_name}_{extension.lstrip('.')}.csv"
        else:
            output_csv_path = Path(output_csv_path).resolve()
            output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Конвертируем в CSV
        convert_annotation_to_csv(annotation, str(output_csv_path))
        
    finally:
        # Возвращаемся в исходную директорию
        os.chdir(original_dir)
    
    return str(output_csv_path)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Использование:")
        print("  python wfdb_to_csv.py <input_folder> [output_folder]")
        print("\nПримеры:")
        print("  python wfdb_to_csv.py /data/mitdb")
        print("  python wfdb_to_csv.py /data/mitdb /output/csv")
        sys.exit(1)
    
    input_folder = sys.argv[1]
    output_folder = sys.argv[2] if len(sys.argv) > 2 else None
    
    convert_folder_annotations(input_folder, output_folder)
