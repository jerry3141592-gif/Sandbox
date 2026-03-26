#!/usr/bin/env python3
"""
Скрипт для чтения разметки WFDB и подсчета количества размеченных событий
по всем файлам в указанной папке.

WFDB (Waveform Database) - формат для хранения физиологических сигналов и аннотаций.
"""

import os
import argparse
from pathlib import Path
from collections import defaultdict
import wfdb


def count_annotations_in_record(record_path):
    """
    Подсчитать количество аннотаций в одной записи WFDB.
    
    Args:
        record_path: Путь к записи WFDB (без расширения .dat, .hea, .atr)
    
    Returns:
        dict: Словарь с количеством аннотаций по типам
        int: Общее количество аннотаций
    """
    try:
        # Читаем аннотации из файла
        annotation = wfdb.rdann(record_path, 'atr')
        
        # Подсчитываем аннотации по типам
        ann_counts = defaultdict(int)
        for symbol in annotation.symbol:
            ann_counts[symbol] += 1
        
        total_count = len(annotation.symbol)
        return dict(ann_counts), total_count
    
    except FileNotFoundError:
        print(f"  Предупреждение: Файл аннотаций не найден для {record_path}")
        return {}, 0
    except Exception as e:
        print(f"  Ошибка при чтении {record_path}: {e}")
        return {}, 0


def find_wfdb_records(folder_path):
    """
    Найти все записи WFDB в указанной папке.
    
    Записи WFDB состоят из файлов .hea (заголовок), .dat (данные), .atr (аннотации).
    Ищем по файлам .hea, так как они обязательны для каждой записи.
    
    Args:
        folder_path: Путь к папке с данными
    
    Returns:
        list: Список путей к записям WFDB (без расширений)
    """
    records = []
    folder = Path(folder_path)
    
    if not folder.exists():
        raise FileNotFoundError(f"Папка не найдена: {folder_path}")
    
    # Ищем все файлы .hea
    hea_files = list(folder.glob("*.hea"))
    
    for hea_file in hea_files:
        # Извлекаем имя записи без расширения
        record_name = hea_file.stem
        record_path = str(hea_file.parent / record_name)
        records.append(record_path)
    
    # Также ищем в подпапках (рекурсивно)
    for subfolder in folder.rglob("*"):
        if subfolder.is_dir():
            sub_hea_files = list(subfolder.glob("*.hea"))
            for hea_file in sub_hea_files:
                record_name = hea_file.stem
                record_path = str(hea_file.parent / record_name)
                if record_path not in records:
                    records.append(record_path)
    
    return sorted(records)


def process_folder(folder_path, verbose=True):
    """
    Обработать все записи WFDB в папке и подсчитать аннотации.
    
    Args:
        folder_path: Путь к папке с данными WFDB
        verbose: Выводить подробную информацию
    
    Returns:
        dict: Общая статистика по всем файлам
    """
    print(f"\nПоиск записей WFDB в папке: {folder_path}")
    
    records = find_wfdb_records(folder_path)
    
    if not records:
        print("Записи WFDB не найдены.")
        return {}
    
    print(f"Найдено записей: {len(records)}\n")
    
    # Общая статистика
    total_annotations = 0
    global_ann_counts = defaultdict(int)
    file_stats = []
    
    for record_path in records:
        if verbose:
            print(f"Обработка: {record_path}")
        
        ann_counts, count = count_annotations_in_record(record_path)
        
        if verbose and count > 0:
            print(f"  Найдено аннотаций: {count}")
            for ann_type, cnt in sorted(ann_counts.items()):
                print(f"    {ann_type}: {cnt}")
        
        total_annotations += count
        for ann_type, cnt in ann_counts.items():
            global_ann_counts[ann_type] += cnt
        
        file_stats.append({
            'record': record_path,
            'count': count,
            'annotations': ann_counts
        })
    
    # Вывод итоговой статистики
    print("\n" + "="*60)
    print("ИТОГОВАЯ СТАТИСТИКА")
    print("="*60)
    print(f"Всего обработано файлов: {len(records)}")
    print(f"Общее количество аннотаций: {total_annotations}")
    
    if global_ann_counts:
        print("\nРаспределение по типам аннотаций:")
        for ann_type, cnt in sorted(global_ann_counts.items()):
            print(f"  {ann_type}: {cnt}")
    
    print("="*60)
    
    return {
        'total_files': len(records),
        'total_annotations': total_annotations,
        'annotation_counts': dict(global_ann_counts),
        'file_stats': file_stats
    }


def main():
    parser = argparse.ArgumentParser(
        description='Подсчет количества размеченных событий в файлах WFDB',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python count_wfdb_annotations.py /path/to/wfdb/folder
  python count_wfdb_annotations.py ./data --quiet
  python count_wfdb_annotations.py /data/mitdb -o results.json
        """
    )
    
    parser.add_argument(
        'folder',
        help='Путь к папке с файлами WFDB'
    )
    
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Не выводить подробную информацию о каждом файле'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Сохранить результаты в JSON файл'
    )
    
    args = parser.parse_args()
    
    # Обрабатываем папку
    results = process_folder(args.folder, verbose=not args.quiet)
    
    # Сохраняем результаты в JSON если указано
    if args.output:
        import json
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nРезультаты сохранены в: {args.output}")
    
    return results


if __name__ == '__main__':
    main()
