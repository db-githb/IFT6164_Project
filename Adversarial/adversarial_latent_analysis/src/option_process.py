import json
from constant import exec_args, option_types, bool_conversion_dict

def load_json(path):
    data = []
    with open(path, 'r') as fp:
        try:
            data = json.load(fp)
            fp.close()
        except json.decoder.JSONDecodeError:
            print('The json file should not be empty')
            exit(1)
    return data

def option_value_convert(value, type):
    converted_value = None
    if type == 'str':
        converted_value = str(value)
    elif type == 'int':
        converted_value = int(value)
    elif type == 'float':
        converted_value = float(value)
    elif type == 'bool':
        converted_value = bool_conversion_dict[value]
    return converted_value

def process_options(options_path, display=False):
    options_json = load_json(options_path)
    for option_data_line in options_json:
        option_name = option_data_line[0]
        option_type = option_data_line[1]
        if option_type != option_types[-2] and option_type != option_types[-1]:
            # The option is not a list or a dict
            exec_args[option_name] = option_value_convert(option_data_line[2], option_type)
        elif option_type == option_types[-2]:
            key_type = option_data_line[2]
            value_type = option_data_line[3]
            converted_key = option_value_convert(option_data_line[4], key_type)
            # Creation of variable if it does not exists
            if option_name not in exec_args:
                exec_args[option_name] = {}
            if len(option_data_line[5:]) > 1:
                exec_args[option_name][converted_key] = [option_value_convert(elem, value_type) for elem in option_data_line[5:]]
            else:
                exec_args[option_name][converted_key] = option_value_convert(option_data_line[5], value_type)
        elif option_type == option_types[-1]:
            value_type = option_data_line[2]
            if option_name not in exec_args:
                exec_args[option_name] = []
            for val in option_data_line[3:]:
                exec_args[option_name].append(option_value_convert(val, value_type)) 
    if display:
        print('The execution arguments are:')
        print(exec_args)
        