import sys
import os
import questionary

from modules.loss.hardware_module import hardware_module
from utils.hardware_utils import hw_visualizzer, hw_test_all, hw_choose_specific, add_hw_config
from components.colors import colors
from utils.model_utils_tf2torch import load_trained_model, check_model_path

# Ensure the custom_hardware_module is correctly imported
hw_mod = hardware_module()
hw_names = list(hw_mod.nvdla.keys())

print(colors.CYAN, "|  ----------- HARDWARE LATENCY TESTER ----------  |\n", colors.ENDC)

# Loading of the pre-trained model
while True:
    model_path = questionary.path("Enter the path of the pre-trained model").ask()
    if not model_path:
        model_path = "gesture/dashboard/model/model.keras"  # Default path if not provided
        break
    if check_model_path(model_path):
        break

print(f"{colors.OKGREEN}Loading and converting model: {os.path.basename(model_path)}{colors.ENDC}")

# Load and convert the model
model = load_trained_model(model_path)
if model is None:
    print(colors.FAIL, "Failed to load and convert the model.", colors.ENDC)
    sys.exit(1)

print(f"{colors.OKGREEN}Model successfully loaded and converted to PyTorch!{colors.ENDC}")

# Menu for the API
print(colors.CYAN, "|  ----------- HARDWARE LATENCY TESTER ----------  |\n", colors.ENDC)
while True:
    choice = questionary.select(
        "Choose an option:",
        choices=[
            "1: View available hardware configurations",
            "2: Test all hardware configurations",
            "3: Choose specific configurations to test",
            "4: Add a new hardware configuration",
            "5: Exit"],
    ).ask()

    if choice.startswith("1"):
        print(colors.OKBLUE, "|  ----------- AVAILABLE HARDWARE  ----------  |\n", colors.ENDC)
        hw_visualizzer(hw_mod)
        print(colors.OKBLUE, "|  ------------------------------------  |\n", colors.ENDC)
    elif choice.startswith("2"):
        hw_test_all(hw_mod, model)
    elif choice.startswith("3"):
        hw_choose_specific(hw_mod, model)
    elif choice.startswith("4"):
        add_hw_config()
        hw_mod = hardware_module()
        hw_names = list(hw_mod.nvdla.keys())
    elif choice.startswith("5"):
        sys.exit()
    else:
        print(colors.FAIL, "Invalid choice. Please try again.", colors.ENDC)