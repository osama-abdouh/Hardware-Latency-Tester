import questionary
import os
from components.colors import colors
from modules.loss.hardware_module import hardware_module
from menus.hardware_menu import HardwareMenu
from utils.hardware_utils import select_hw_config
from utils.model_utils import check_if_path_is_model
from utils.testing_utils import single_model_test, multi_model_test

class TestingMenu:
    def __init__(self):
        self.hw_mod = hardware_module()
        self.hardware_menu = HardwareMenu(self.hw_mod)

    def display_header(self):
        print(colors.CYAN + "+-----------------------------------+")
        print("|       Model Testing Menu        |")
        print("+-----------------------------------+" + colors.ENDC)

    def print_grouped_results(self, results):
        # Function to print grouped results
        grouped = {}
        for res in results:
            model = res.get("model", "")
            if model not in grouped:
                grouped[model] = []
            grouped[model].append(res)

        print(colors.OKGREEN, "|  ---------------------- TEST ----------------------  |", colors.ENDC)
        for model, items in grouped.items():
            print(colors.MAGENTA, f"Model: {model}", colors.ENDC)
            for r in items:
                hw = r.get("HW config")
                lat = r.get("latency(s)")
                print(f"Configuration: {hw} | Latency: {lat:.6f} seconds")
        print(colors.OKGREEN, "|  --------------------------------------------------  |\n", colors.ENDC)

    def test_hw_latency(self):
        hw_choose = select_hw_config(self.hw_mod)
        if not hw_choose:
            return  # Exit if no hardware configuration is available

        results = None
        while True:
            models_path=questionary.path("Enter the model path or directory path containing multiple models").ask()

            if not models_path:
                models_path = "gesture/dashboard/model" #Default path if not provided
                print("Using default model(s) path:", models_path)
                break

            models_path = os.path.abspath(os.path.expanduser(models_path))

            if check_if_path_is_model(models_path):
                results = single_model_test(self.hw_mod, hw_choose, models_path)
                break
            elif os.path.isdir(models_path):
                results = multi_model_test(self.hw_mod, hw_choose, models_path)
                break
            else:
                print(colors.FAIL, "Invalid file/directory. Please try again.", colors.ENDC)
                continue
        
        if results is None:
            print(colors.FAIL, "No results returned from model test.", colors.ENDC)
            return
        self.print_grouped_results(results)
        while True:
            confirm = questionary.confirm("Press enter to return to the Latency Menu.").ask()
            if confirm:
                break

    def run(self):
        while True:
            if not self.hw_mod:
                print(colors.FAIL, "No hardware configurations available. Please manage hardware first.", colors.ENDC)
                self.hardware_menu = HardwareMenu(self.hw_mod)  
                
            self.display_header()
            choice = questionary.select(
                "Select an option:",
                choices=[
                    "1: Manage Hardware Configurations",
                    "2: Test Hardware Latency",
                    "3: Back to Main Menu"
                ]
            ).ask()

            if not choice:
                break

            choice_num = choice[0] if choice else ""

            if choice_num == "1":
                self.hardware_menu.run()
                self.hw_mod = self.hardware_menu.hw_mod
            elif choice_num == "2":
                os.system('cls' if os.name == 'nt' else 'clear')
                self.test_hw_latency()
            elif choice_num == "3":
                break