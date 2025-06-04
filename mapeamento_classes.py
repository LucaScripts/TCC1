# Dicionário padrão para mapeamento de classes
# Mapeia o 'Situação (código)' original para um novo índice inteiro (0 a N-1)
# Considera o agrupamento de CAU (3) em CAN (2) e a remoção de classes com < 10 exemplos (NF, TR)

mapping_dict = {
    0: 0,  # CAC -> 0
    1: 1,  # CAI -> 1
    2: 2,  # CAN/CAU -> 2
    4: 3,  # ES -> 3
    5: 4,  # FO -> 4
    6: 5,  # LAC -> 5
    7: 6,  # LFI -> 6
    8: 7,  # LFR -> 7
    9: 8,  # MT -> 8
    10: 9, # NC -> 9
    12: 10 # TF -> 10
}

# Dicionário reverso para interpretação (novo índice para sigla)
reverse_mapping_dict = {
    0: 'CAC',
    1: 'CAI',
    2: 'CAN/CAU',
    3: 'ES',
    4: 'FO',
    5: 'LAC',
    6: 'LFI',
    7: 'LFR',
    8: 'MT',
    9: 'NC',
    10: 'TF'
}

# Lista das siglas na ordem do novo índice
class_names_ordered = [
    'CAC', 'CAI', 'CAN/CAU', 'ES', 'FO', 'LAC', 'LFI', 'LFR', 'MT', 'NC', 'TF'
]

