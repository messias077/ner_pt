def dump_report(rep_dict, output_file):
    if not rep_dict:
        print('\nReport is empty!')
        return
    with open(output_file, 'w') as out_file:
        out_file.write('label,precision,recall,f1-score\n')
        for label in rep_dict.keys():
            if label != 'accuracy':
                out_file.write(f"{label},{rep_dict[label]['precision']},{rep_dict[label]['recall']},"
                               f"{rep_dict[label]['f1-score']}\n")
            else:
                out_file.write(f"{label},,,{rep_dict[label]}\n")


def compute_direct_match(original_ners, predicted_ners):
    if len(original_ners) != len(predicted_ners):
        print("\nERROR: 'original_ners' and 'predicted_ners' must be the same size.")
        return None
    # Contabiliza a quantidade de ners extraídas no dataset original
    qty_original_ners = sum([len(n) for n in original_ners])
    if qty_original_ners == 0:  # Para evitar divisão por 0
        print("\nERROR: 'original_ners' doesn't have ner")
        return None
    matchs = 0
    for i in range(len(original_ners)):
        if original_ners[i] == predicted_ners[i]:  # Conseguiu predizer todas as ners corretamente
            matchs += len(original_ners[i])
        else:  # Errou alguma(s) ou todas ners
            for ner in set(original_ners[i]):
                count_ner_original = original_ners[i].count(ner)
                count_ner_predicted = predicted_ners[i].count(ner)

                if count_ner_original >= count_ner_predicted:
                    matchs += count_ner_predicted
                else:
                    matchs += count_ner_original
    return matchs/qty_original_ners


def generate_test_file(test_results_file, new_test_file):
    with open(new_test_file, 'w+', encoding='utf8') as new_file:
        with open(test_results_file, 'r', encoding='utf8') as file:
            for line in file:
                if line != '\n':
                    line = line.strip()
                    spliter = line.split(' ')
                    token = spliter[0]
                    tag_1 = spliter[1]
                    tag_2 = spliter[2]
                    new_file.write(str(token) + ' ' + str(tag_1) +
                                   ' ' + str(tag_2) + '\n')
                else:
                    new_file.write(line)
