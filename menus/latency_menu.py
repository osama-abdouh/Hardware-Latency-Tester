import questionary

from modules.loss.hardware_module import hardware_module
from components.colors import colors
from menus.hardware_menu import HardwareMenu
from menus.model_test_menu import ModelTestMenu




class LatencyMenu:
    def __init__(self):
        self.hardware_menu = HardwareMenu()
        self.model_test_menu = ModelTestMenu()
        self.hw_mod = hardware_module()

    def display_header(self):
        print(colors.CYAN + "+-----------------------------------+")
        print("|       Latency Testing Menu        |")
        print("+-----------------------------------+" + colors.ENDC)

    def run(self):
        self.display_header()
        while True:
            choice = questionary.select(
                "Select an option:",
                choices=[
                    "1: Manage Hardware Configurations",
                    "2: Test Latency",
                    "3: Back to Main Menu"
                ]
            ).ask()

            if not choice:
                break

            choice_num = choice[0] if choice else ""

            if choice_num == "1":
                self.hardware_menu.run()
                self.hardware_menu.hw_mod = self.hw_mod
            elif choice_num == "2":
                self.model_test_menu.run()
            elif choice_num == "3":
                break