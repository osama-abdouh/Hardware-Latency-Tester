import os
import json

from components.colors import colors
import questionary

# create json configuration and/or 
def load_or_create_nvdla_configs(path="nvdla/nvdla_configs.json"):
    if not os.path.exists(path):
        with open(path, 'w') as f:
            json.dump([], f, indent=4)
    with open(path, 'r') as f:
        return json.load(f)
    

# list of available hardware
def hw_visualizzer(hw_mod):
    for config_name, config in hw_mod.nvdla.items():
        print(f"- {config_name} | Path: {config['path']} | Cost: {config['cost']:.2f}")

# Add a new configuration to the available configurations
def add_hw_config():
    nvdla_list = load_or_create_nvdla_configs()

    path=questionary.path("Enter the path of the hardware configuration to add:").ask()
    if not path or not os.path.exists(path) or not path.endswith(".yaml"):
        print(colors.FAIL, "Invalid path or file does not exist.", colors.ENDC)
        return
    
    while True:
        name=questionary.text("Enter the name of the hardware configuration:").ask()
        if any(cfg["name"] == name for cfg in nvdla_list):
            print(colors.FAIL, "The name already exists.", colors.ENDC)
        else:
            break
    try: 
        area = float(questionary.text("Enter the Area (mm²):").ask())
        cost_par = float(questionary.text("Enter the cost per mm²:").ask())
    except ValueError:
        print(colors.FAIL, "Area and cost must be numeric values.", colors.ENDC)
        return
    
    new_config = {
        "name": name,
        "path": os.path.basename(path),
        "area": area,
        "C/mm2": cost_par
    }

    nvdla_list.append(new_config)

    with open("nvdla/nvdla_configs.json", 'w') as f:
        json.dump(nvdla_list, f, indent=2)
    print(colors.OKGREEN,f"Added configuration: {new_config['name']}", colors.ENDC)

def select_hw_config(hw_mod):
    print(colors.OKGREEN, "|  ----------- SELECT HARDWARE CONFIGURATION ----------  |\n", colors.ENDC)
    hw_choices = questionary.checkbox(
        "Select the configurations to test:",
        choices=hw_mod.nvdla.keys()
    ).ask()
    return hw_choices

def hw_latency_test(hw_mod, model):
    hw_choose = select_hw_config(hw_mod)

    print(colors.OKBLUE, "|  ----------- HARDWARE TESTED ----------  |\n", colors.ENDC)
    for hw_c in hw_choose:
        if hw_c in hw_mod.nvdla:
            latency = hw_mod.get_model_latency(model, hw_mod.nvdla[hw_c]['path'])
            print(f"Configuration: {hw_c} | Latency: {latency/10**9:.6f} secondi")
    print(colors.OKBLUE, "|  ------------------------------------  |\n", colors.ENDC)

# display available hardware configurations
def hw_test_all(hw_mod, model):
    print(colors.OKBLUE, "|  ----------- HARDWARE TESTED ----------  |\n", colors.ENDC)
    for config_name, config in hw_mod.nvdla.items():
        latency = hw_mod.get_model_latency(model, config['path'])
        print(f"Configuration: {config_name} | Latency: {latency/(10**9):.6f} secondi")
    print(colors.OKBLUE, "|  ------------------------------------  |\n", colors.ENDC)

# choose of the specific hardwares configurations to test

