import questionary
from modules.loss.hardware_module import hardware_module
from components.colors import colors
from hardware_utils import hw_visualizzer, add_hw_config

class HardwareMenu:
    def __init__(self, hw_mod=None):
        self.hw_mod = hw_mod if hw_mod else hardware_module()
        pass

    def display_header(self):
        print(colors.CYAN + "+-----------------------------------+")
        print("|       Hardware Management Menu    |")
        print("+-----------------------------------+" + colors.ENDC)
    
    def view_hardware_configurations(self):
        if not self.hw_mod:
            print(colors.FAIL + "Hardware module not initialized!" + colors.ENDC)
            return
            
        print(colors.OKBLUE + "+----------- AVAILABLE HARDWARE -----------+" + colors.ENDC)  
        hw_visualizzer(self.hw_mod)
        print(colors.OKBLUE + "+------------------------------------------+" + colors.ENDC)

    def add_hardware_configuration(self):
        if not self.hw_mod:
            print(colors.FAIL + "Hardware module not initialized!" + colors.ENDC)
            return

        print(colors.OKBLUE + "+----------- ADD NEW HARDWARE -----------+" + colors.ENDC)
        add_hw_config()
        print(colors.OKBLUE + "+------------------------------------------+" + colors.ENDC)

    def remove_hardware_configuration(self):
        print(colors.OKBLUE + "+----------- REMOVE HARDWARE -----------+" + colors.ENDC)
        # Implementation for removing hardware configuration goes here
        print(colors.OKBLUE + "+------------------------------------------+" + colors.ENDC)

    def run(self):
        self.display_header()
        while True:
            choice = questionary.select(
                "Select an option:",
                choices=[
                    "1: View Hardware Configurations",
                    "2: Add New Hardware",
                    "3: Remove Hardware",
                    "4: Back"
                ]
            ).ask()

            if not choice:
                break

            choice_num = choice[0] if choice else ""

            if choice_num == "1":
                self.view_hardware_configurations()
            elif choice_num == "2":
                self.add_hardware_configuration()
            elif choice_num == "3":
                print("Removing Hardware...")
            elif choice_num == "4":
                break