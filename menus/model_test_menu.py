import questionary
from components.colors import colors
from model_utils import load_trained_model, check_model_path
from hardware_utils import hw_choose_specific

class ModelTestMenu:

    def __init__(self, hw_mod=None):
        self.hw_mod = hw_mod

    def display_header(self):
        print(colors.OKBLUE + "+-----------------------------------+")
        print("|         Model Test Menu           |")
        print("+-----------------------------------+" + colors.ENDC)

    def single_model_test(self):
        while True:
            model_path=questionary.path("Enter the path of the pre-trained model").ask()
            
            if not model_path:
                model_path = "gesture/dashboard/model/model.keras" #Default path if not provided
                print("Using default model path:", model_path)
                break

            if check_model_path(model_path) == True:
                break

        model=load_trained_model(model_path)
        print(colors.OKGREEN, "Model loaded successfully", colors.ENDC)

        hw_choose_specific(self.hw_mod, model)

    def multi_model_test(self):
        print("Multi-model testing is not yet implemented.")

    def run(self):
        self.display_header()
        while True:
            choice = questionary.select(
                "Select an option:",
                choices=[
                    "1: Test single model",
                    "2: Test multiple models",
                    "3: Back to Latency Menu"
                ]
            ).ask()

            if not choice:
                break

            choice_num = choice[0] if choice else ""

            if choice_num == "1":
                self.single_model_test()
            elif choice_num == "2":
                self.multi_model_test()
            elif choice_num == "3":
                break

