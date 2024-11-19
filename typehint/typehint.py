import os
import sys
import subprocess
import re

def main():
    typehint_include = '#include "typehint.h"\n'
    extra_flags = ['-fsyntax-only', '-Wunused']
    typehint_id_prefix = '_typehint_id'
    typehint_trigger = "typehint"
    typehint_print_val_type = "TYPEHINT_PRINT_VAL_TYPE"
    typehint_print_using_type = "TYPEHINT_PRINT_USING_TYPE"
    typehing_type_passthrough = "TYPEHINT_TYPE_PASSTHROUGH"
    temp_file_prefix = "_typehint_temp_"

    if len(sys.argv) < 3:
        print("Usage: python typehint.py <compiler> [compiler_flags] <source_file>")
        sys.exit(1)

    source_file = sys.argv[-1]
    temp_file = os.path.join(os.path.dirname(source_file), temp_file_prefix + os.path.basename(source_file))
    compiler_command = sys.argv[1:-1] + extra_flags

    with open(source_file, 'r') as f:
        lines = f.readlines()

    id_counter = 0
    id_map = {}
    version1_lines = []
    version2_lines = []

    for line in lines:
        if f'// {typehint_trigger}' in line:
            id_counter += 1
            id_str = f'{typehint_id_prefix}:{id_counter}'
            id_map[id_str] = []

            before_comment = line.split('//')[0]

            line_v1 = before_comment + f'// {id_str}\n'
            version1_lines.append(line_v1)

            before_eq, sep, after_eq = before_comment.partition('=')
            if sep:
                expr = after_eq.strip().rstrip(';')

                if before_eq.strip().startswith('using'):
                    print_fn = typehint_print_using_type
                    expr = f"decltype({typehing_type_passthrough}<{expr}>{{}})::T"
                else:
                    print_fn = typehint_print_val_type
                    expr = f"({expr})"

                line_v2 = f"{before_eq}= {print_fn}(\"{id_str}\", {expr});\n"
                version2_lines.append(line_v2)
            else:
                version2_lines.append(line)
        else:
            version1_lines.append(line)
            version2_lines.append(line)


    version2_lines.insert(0, typehint_include)

    with open(temp_file, 'w') as f:
        f.writelines(version2_lines)

    result = subprocess.run(compiler_command + [temp_file], capture_output=True, text=True)
    os.remove(temp_file)

    if result.returncode == 0:
        warnings = result.stderr
    else:
        print("Compilation failed:")
        print(result.stderr)
        sys.exit(1)

    for line in warnings.splitlines():
        # example match from Apple clang version 16.0.0
        # inline.cpp:11:14: note: in instantiation of function template specialization 'typehint::static_print<StringLiteral<5>{"<id_str>"}, StringLiteral<11>{"<type>"}>' requested here
        match = re.search(r'<StringLiteral<\d+>{"(.+)"}, StringLiteral<\d+>{"(.+)"}>', line.replace('typehint::StringLiteral', 'StringLiteral'))

        if match:
            id_str, type_str = match.groups()
            id_map[id_str].append(type_str)

    for line in version1_lines:
        updated_lines = []
        for line in version1_lines:
            id_regex = f"{typehint_id_prefix}:\\d+"
            match = re.search(id_regex, line)

            if match:
                id_str = match.group(0)
                type_list = id_map.get(id_str) or ["Untraced"]
                type_list_str = " ".join([f"[{t}]" for t in type_list])
                line = re.sub(id_regex, f'{typehint_trigger}: {type_list_str}', line)

            updated_lines.append(line)

    with open(source_file, 'w') as f:
        f.writelines(updated_lines)

if __name__ == '__main__':
    main()