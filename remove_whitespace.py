with open('noise_text.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()

non_empty_lines = [line for line in lines if line.strip()]
print(len(non_empty_lines))

with open('noise_text.txt', 'w', encoding='utf-8') as file:
    file.writelines(non_empty_lines)
