import os
import json

from components.colors import colors
import questionary
from utils.model_utils import load_trained_model
from tqdm import tqdm

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
    print(colors.CYAN, "|  ----------- SELECT HARDWARE CONFIGURATION ----------  |", colors.ENDC)
    while True:
        hw_choices = questionary.checkbox(
            "Select the HW configurations to test:",
            choices=hw_mod.nvdla.keys()
        ).ask()
        if hw_choices == []:
            print(colors.FAIL, "You must select at least one configuration.", colors.ENDC)
        else:
            break
    print(colors.CYAN, "|  --------------------------------------------------  |\n", colors.ENDC)
    return hw_choices

def remove_hw_config(hw_mod):
    nvdla=load_or_create_nvdla_configs()
    if not nvdla:
        print(colors.FAIL, "No hardware configurations available to remove.", colors.ENDC)
        return
    print(colors.OKGREEN, "|  ----------- DELETE HARDWARE CONFIGURATION ----------  |", colors.ENDC)

    hw_choices = select_hw_config(hw_mod)

    confirm = questionary.confirm(
        f"Are you sure you want to remove the selected configurations: {', '.join(hw_choices)}?",
        default=False
    ).ask()

    if not confirm:
        print(colors.FAIL, "Operation cancelled by user.", colors.ENDC)
        return

    new_list = [cfg for cfg in nvdla if cfg.get("name") not in hw_choices]
    with open("nvdla/nvdla_configs.json", 'w') as f:
        json.dump(new_list, f, indent=2)

    if hw_mod and hasattr(hw_mod, 'nvdla'):
        for name in hw_choices:
            if name in hw_mod.nvdla:
                try:
                    del hw_mod.nvdla[name]
                except KeyError as e:
                    pass

    print(colors.OKGREEN, f"Deleted: {', '.join(hw_choices)}", colors.ENDC)
    return hw_choices