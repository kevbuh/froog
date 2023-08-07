import os

def count_lines_of_code(folder_path, file_extension='.py'):
  total_lines = 0
  for root, _, files in os.walk(folder_path):
    for file_name in files:
      if file_name.endswith(file_extension):
        file_path = os.path.join(root, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
          lines = file.readlines()
          total_lines += len(lines)
  return total_lines

if __name__ == "__main__":
  folder_path = "./ribbit"
  lines_of_code = count_lines_of_code(folder_path)

  print(f"Total lines of code in the folder: {lines_of_code}")