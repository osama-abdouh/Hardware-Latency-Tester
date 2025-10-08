import questionary
import sys
from components.colors import colors
from menus.latency_menu import LatencyMenu
from menus.conversion_menu import ConversionMenu
from menus.training_menu import TrainingMenu

class MainMenu:
    def __init__(self):
        self.latency_menu = LatencyMenu()
        self.conversion_menu = ConversionMenu()
        self.training_menu = TrainingMenu()

    def display_header(self):
        print(colors.CYAN + "+-----------------------------------+")
        print("|         ModelBench                |")
        print("+-----------------------------------+" + colors.ENDC)

    def run(self):
        self.display_header()

        while True:
            choice = questionary.select(
                "Select a tool:",
                choices=[
                    "1: Latency testing",
                    "2: Model conversion",
                    "3: Model training",
                    "4: Exit"
                ]
            ).ask()
        
            if not choice:
                break

            choice_num = choice[0] if choice else ""

            if choice_num == "1":
                self.latency_menu.run()
            elif choice_num == "2":
                self.conversion_menu.run()
            elif choice_num == "3":
                self.training_menu.run()
            elif choice_num == "4":
                print("Goodbye!")
                sys.exit(0)
            else:
                print(colors.FAIL, "Invalid choice. Please try again.", colors.ENDC)